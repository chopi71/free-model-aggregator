// The trial brief, the providers' own catalogs, and the catalog this entry registers.
//
// The brief at $MAKEFUN_PROMPT_FILE is prose carrying three JSON documents, found here by their
// `schema` member: the provider directory (ten free-tier providers: id, wire, auth scheme, key
// variable, key, base path), the quality table (every free model's rating and capabilities, and
// each request's best rating), and the workload manifest (the twenty requests in order, with the
// requirements a model must meet to serve each).
//
// The registered catalog is built from the providers' own catalog routes, probed in each
// provider's wire, because the providers' listings are the truth the catalog is scored against:
// they carry the paid models the brief never lists, and each model's free flag and capabilities.
// A probe that fails is retried, then its provider is left out rather than guessed.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** The base path the catalog reports (`/p/<id>`), or an absolute base URL when the brief gives one. */
  base: string;
  key_env: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: WorkloadRequest[];
  error: string | null;
}

/** One model as a provider's catalog route listed it; null where the listing said nothing. */
export interface ListedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ListedModel[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** A base path joined to the env's origin; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider's auth scheme wants its key in; `none` and an empty key send nothing. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '' || auth === 'none') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a document on its own: keep looking.
    }
  }
  return out;
}

/** The index of the `}` closing the object opened at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base = str(v.path) !== '' ? str(v.path) : str(v.base_url) !== '' ? str(v.base_url) : str(v.base);
  return { id, name: str(v.name) !== '' ? str(v.name) : id, wire, auth, base, key_env: str(v.key_env), key: str(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = str(v.model);
  const provider = str(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: numOrNull(v.rating_milli) ?? 0,
    context_length: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: typeof v.weight === 'number' && Number.isFinite(v.weight) ? v.weight : 1,
    requires: {
      tools: req.tools === true,
      vision: req.vision === true,
      json: req.json === true,
      ...(numOrNull(req.min_context) !== null ? { min_context: numOrNull(req.min_context) as number } : {}),
    },
  };
}

/** The brief at `path`: its three documents, found by schema. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory === undefined) return { ...empty, error: `the brief holds no ${PROVIDERS_SCHEMA} document` };
  const providers = (directory.providers as unknown[]).flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings = ratingsDoc === undefined ? [] : ((ratingsDoc.ratings as unknown[]).flatMap((v) => {
    const r = ratingOf(v);
    return r === null ? [] : [r];
  }));
  const best = new Map<string, number>();
  if (ratingsDoc !== undefined && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best as unknown[]) {
      if (isRecord(v) && str(v.id) !== '') best.set(str(v.id), numOrNull(v.rating_milli) ?? 0);
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest = workloadDoc === undefined ? [] : ((workloadDoc.manifest as unknown[]).flatMap((v) => {
    const w = workloadOf(v);
    return w === null ? [] : [w];
  }));
  return { providers, ratings, best, manifest, error: null };
}

/** The quality table's row for one provider's model, when it has one. */
export function ratingRow(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** A listing's price block read as a free flag: true when every price is zero, null when unpriced. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function listedModel(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? str(v.name).replace(/^models\//, '') : str(v.id);
  if (id === '') return null;
  return {
    id,
    display_name: str(v.display_name) !== '' ? str(v.display_name) : str(v.displayName) !== '' ? str(v.displayName) : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

function listedModels(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModel(wire, v);
    return m === null ? [] : [m];
  });
}

/**
 * Probe one provider's catalog route in its own wire, retrying a failed probe twice: the catalog
 * is scored on every provider, so one transient refusal should not erase a provider from it.
 * Catalog routes name no model, so a probe spends none of a provider's inference budget.
 */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
      if (!res.ok) continue;
      const body: unknown = await res.json();
      return { provider, ok: true, models: listedModels(provider.wire, body) };
    } catch {
      // A refused or unreachable catalog route: retry, then leave the provider out.
    }
  }
  return { provider, ok: false, models: [] };
}

/** Whether a listed model is free: the quality table's row where it has one, else the listing's word. */
export function modelFree(brief: Brief, providerId: string, model: ListedModel): boolean {
  const row = ratingRow(brief, providerId, model.id);
  if (row !== null) return row.free;
  return model.free ?? false;
}

/**
 * The registered catalog: every provider that answered its probe, in the brief's order, its
 * models sorted by id, each with the free flag and context length the listing and the quality
 * table agree on. Its base_url is the directory's own path, never joined to the env's origin:
 * the origin differs between a live trial and its replay, and the document must not.
 */
export function buildCatalog(brief: Brief, probes: readonly Probe[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRow(brief, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: modelFree(brief, probe.provider.id, model),
        context_length: row?.context_length ?? model.context_length ?? null,
      });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}
