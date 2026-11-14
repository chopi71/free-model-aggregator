// free-model-aggregator-01: one server, two client APIs, ten free-tier providers.
//
// The server reads the trial's brief (the provider directory, the sealed quality table and the
// workload manifest), probes each provider's own catalog route, and registers the catalog it
// built. For every request the client sends — OpenAI Chat Completions or Anthropic Messages, in
// JSON or in server-sent events — it ranks every free model that meets the request's
// requirements by the quality table's rating, speaks the best one's wire (translating the
// request and the answer where the wires differ), and when a provider refuses, counts the
// refusal and answers from the next model down the ranking. A provider whose refusal says its
// budget is spent or that it is out is not asked again: neither heals inside a trial. A refusal
// on the provider's own schedule is transient and costs only that one call.
//
// Every choice is a function of the brief, the requests and the providers' answers so far —
// no wall clock and no randomness anywhere in a decision — so a replayed trial, which sees the
// same requests and the same provider answers, is served the same way.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingRow,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type ListedModel,
  type Probe,
  type Requirement,
} from './lib/brief.ts';
import {
  answerFromAnthropic,
  answerFromGemini,
  answerFromOpenAI,
  canonFromAnthropic,
  canonFromOpenAI,
  parseSse,
  refusalCause,
  renderAnthropic,
  renderAnthropicAnswer,
  renderAnthropicStream,
  renderGemini,
  renderOpenAI,
  renderOpenAIAnswer,
  renderOpenAIStream,
  FIXED_CREATED,
  type CanonAnswer,
  type CanonRequest,
} from './lib/xlate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most upstream calls one client request may cost before the server answers with an error.
 * Twenty-six free models meet most requests, so the cap binds only when providers refuse in a
 * run; ten attempts leave room for a whole tier of refusals and still answer inside the
 * client's patience. */
const MAX_ATTEMPTS = 10;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** Providers whose budget is spent or whose outage has begun: they refuse every later call. */
const deadProviders = new Set<string>();
/** How many workload requests have arrived: the manifest is their order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider (in parallel; the catalog's order is the brief's, not the probes') and build it. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  catalog = buildCatalog(brief, probes);
}

/** One routable model: its provider's probe, its listing, its rating and its true capabilities. */
interface Candidate {
  probe: Probe;
  model: ListedModel;
  rating: number;
  providerIndex: number;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  contextLength: number | null;
}

function candidates(): Candidate[] {
  const out: Candidate[] = [];
  probes.forEach((probe, providerIndex) => {
    if (!probe.ok) return;
    for (const model of probe.models) {
      if (!modelFree(brief, probe.provider.id, model)) continue;
      const row = ratingRow(brief, probe.provider.id, model.id);
      out.push({
        probe,
        model,
        rating: row?.rating_milli ?? 0,
        providerIndex,
        tools: model.tools ?? row?.tools ?? null,
        vision: model.vision ?? row?.vision ?? null,
        json: model.json ?? row?.json ?? null,
        contextLength: model.context_length ?? row?.context_length ?? null,
      });
    }
  });
  return out;
}

function meets(candidate: Candidate, requires: Requirement): boolean {
  if (requires.tools === true && candidate.tools !== true) return false;
  if (requires.vision === true && candidate.vision !== true) return false;
  if (requires.json === true && candidate.json !== true) return false;
  if (requires.min_context !== undefined && (candidate.contextLength === null || candidate.contextLength < requires.min_context)) return false;
  return true;
}

/**
 * The candidates for one request, best first: rating, then the client's own wire (a pass-through
 * risks nothing in translation), then the brief's provider order, then the model id. A JSON-mode
 * OpenAI request narrows to the wires that can carry response_format — OpenAI itself, and Gemini
 * through responseMimeType; an Anthropic-wire provider parses no JSON mode and echoes it absent.
 * Dead providers are left out entirely.
 */
function rankedCandidates(clientApi: 'openai' | 'anthropic', requires: Requirement, jsonMode: boolean): Candidate[] {
  let list = candidates().filter((c) => !deadProviders.has(c.probe.provider.id) && meets(c, requires));
  if (jsonMode && clientApi === 'openai') {
    const carrying = list.filter((c) => c.probe.provider.wire === 'openai' || c.probe.provider.wire === 'gemini');
    if (carrying.length > 0) list = carrying;
  }
  const wireOf = (c: Candidate): number => (c.probe.provider.wire === clientApi ? 0 : 1);
  return list.sort(
    (a, b) =>
      b.rating - a.rating ||
      wireOf(a) - wireOf(b) ||
      a.providerIndex - b.providerIndex ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(canon: CanonRequest): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return {
    tools: canon.tools.length > 0,
    vision: canon.turns.some((t) => t.images.length > 0),
    json: canon.jsonMode,
  };
}

/** One upstream attempt's outcome: the canonical answer, or the refusal's cause. */
type Attempt = { ok: true; answer: CanonAnswer } | { ok: false; cause: 'budget' | 'outage' | 'transient' | 'error' };

/** Call one candidate's provider, in its wire, and read its answer into the canonical answer. */
async function callCandidate(candidate: Candidate, canon: CanonRequest, clientVersion: string | null): Promise<Attempt> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = renderOpenAI(canon, candidate.model.id);
    if (canon.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    payload = renderAnthropic(canon, candidate.model.id);
    headers['anthropic-version'] = clientVersion ?? '2023-06-01';
    if (canon.stream) headers.accept = 'text/event-stream';
  } else {
    const verb = canon.stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}${canon.stream ? '?alt=sse' : ''}`;
    payload = renderGemini(canon);
    if (canon.stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return { ok: false, cause: 'error' };
  }
  const raw = await res.text();
  if (!res.ok) return { ok: false, cause: refusalCause(res.status, raw) };
  const streamed = canon.stream || (res.headers.get('content-type') ?? '').includes('text/event-stream');
  if (provider.wire === 'openai') {
    if (streamed) return { ok: true, answer: answerFromOpenAI(null, parseSse(raw).frames) };
    try {
      return { ok: true, answer: answerFromOpenAI(JSON.parse(raw) as unknown, null) };
    } catch {
      return { ok: false, cause: 'error' };
    }
  }
  if (provider.wire === 'anthropic') {
    if (streamed) return { ok: true, answer: answerFromAnthropic(null, parseSse(raw).events) };
    try {
      return { ok: true, answer: answerFromAnthropic(JSON.parse(raw) as unknown, null) };
    } catch {
      return { ok: false, cause: 'error' };
    }
  }
  if (streamed) return { ok: true, answer: answerFromGemini(null, parseSse(raw).frames) };
  try {
    return { ok: true, answer: answerFromGemini(JSON.parse(raw) as unknown, null) };
  } catch {
    return { ok: false, cause: 'error' };
  }
}

/** Serve one client request: walk the ranking until a provider answers, then render its answer back. */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  clientVersion: string | null,
): Promise<{ status: number; contentType: string; text: string }> {
  const canon = clientApi === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
  const requires = requirementsFor(canon);
  const ranked = rankedCandidates(clientApi, requires, canon.jsonMode);
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : null;
  let attempts = 0;
  for (const candidate of ranked) {
    if (attempts >= MAX_ATTEMPTS) break;
    attempts += 1;
    const attempt = await callCandidate(candidate, canon, clientVersion);
    if (!attempt.ok) {
      if (attempt.cause === 'budget' || attempt.cause === 'outage') deadProviders.add(candidate.probe.provider.id);
      continue;
    }
    const answer = attempt.answer;
    if (clientApi === 'openai') {
      return canon.stream
        ? { status: 200, contentType: 'text/event-stream', text: renderOpenAIStream(answer, candidate.model.id) }
        : { status: 200, contentType: 'application/json', text: JSON.stringify(renderOpenAIAnswer(answer, candidate.model.id)) };
    }
    const named = requestedModel ?? candidate.model.id;
    return canon.stream
      ? { status: 200, contentType: 'text/event-stream', text: renderAnthropicStream(answer, named) }
      : { status: 200, contentType: 'application/json', text: JSON.stringify(renderAnthropicAnswer(answer, named)) };
  }
  const message = 'no free model that meets this request could be reached';
  return clientApi === 'openai'
    ? { status: 502, contentType: 'application/json', text: JSON.stringify({ error: { message, type: 'api_error', code: null } }) }
    : { status: 502, contentType: 'application/json', text: JSON.stringify({ type: 'error', error: { type: 'api_error', message } }) };
}

/** The free model ids the catalog registered, each once, in the catalog's own order. */
function registeredFreeIds(): { id: string; provider: string; display: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string; display: string }[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      const probe = probes.find((p) => p.provider.id === provider.id);
      const listed = probe?.models.find((m) => m.id === model.id);
      out.push({ id: model.id, provider: provider.id, display: listed?.display_name ?? model.id });
    }
  }
  return out;
}

/** GET /v1/models in the shape of the auth header: x-api-key is Anthropic, anything else OpenAI. */
function modelsReply(req: IncomingMessage): Json {
  const ids = registeredFreeIds();
  if (typeof req.headers['x-api-key'] === 'string') {
    return {
      data: ids.map((m) => ({ type: 'model', id: m.id, display_name: m.display, created_at: '2026-01-01T00:00:00Z' })),
      has_more: false,
    };
  }
  return {
    object: 'list',
    data: ids.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function send(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    send(res, 200, 'application/json', JSON.stringify(modelsReply(req)));
    return;
  }
  const clientApi = url.pathname === '/v1/messages' ? 'anthropic' : url.pathname === '/v1/chat/completions' ? 'openai' : null;
  if (req.method === 'POST' && clientApi !== null) {
    const body = await readJson(req);
    if (body === null) {
      const message = 'the request body is not one JSON object';
      send(
        res,
        400,
        'application/json',
        clientApi === 'openai'
          ? JSON.stringify({ error: { message, type: 'invalid_request_error', code: null } })
          : JSON.stringify({ type: 'error', error: { type: 'invalid_request_error', message } }),
      );
      return;
    }
    const version = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : null;
    const reply = await serve(clientApi, body, version);
    send(res, reply.status, reply.contentType, reply.text);
    return;
  }
  const message = `no route ${req.method ?? ''} ${url.pathname}`;
  send(
    res,
    404,
    'application/json',
    url.pathname === '/v1/messages'
      ? JSON.stringify({ type: 'error', error: { type: 'not_found_error', message } })
      : JSON.stringify({ error: { message, type: 'not_found', code: null } }),
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work; exit 0 the moment the env says the workload is done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte, printed and done.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        send(res, 500, 'application/json', JSON.stringify({ error: { message: 'the server failed to answer', type: 'internal_error', code: null } }));
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${address.port} with ${catalog.providers.length} providers and ${registeredFreeIds().length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
