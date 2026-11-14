// One canonical request and answer, and every wire's rendering of them.
//
// The providers answer a request by echoing what they parsed from it — system text, user text,
// tool names, max tokens, stop sequences, images, JSON mode — and the client checks that echo
// against its own request. A translation is therefore right exactly when each wire's parser
// reads the same values out of the rendering as out of the original: this file canonicalises a
// client request once (system text joined the way the parsers join it, each turn's text the way
// the parsers read it), then renders that one form into the OpenAI, Anthropic and Gemini wires.
// Answers take the same road back: a provider's answer, in whichever wire, is read into one
// canonical answer — its whole text and its usage counts — and rendered in the client's wire,
// as one JSON body or as that wire's canonical event stream. Providers in this world answer in
// plain text with the wire's natural stop, so the stop families map stop <-> end_turn <-> STOP.

import { isRecord, type Json } from './brief.ts';

export interface CanonImage {
  mediaType: string;
  data: string | null;
  url: string | null;
}

export interface CanonToolCall {
  id: string;
  name: string;
  /** The call's arguments as one JSON string. */
  args: string;
}

export interface CanonToolResult {
  id: string;
  name: string;
  text: string;
}

export interface CanonTurn {
  role: 'user' | 'assistant';
  /** The turn's text as the wire parsers read it: its text blocks joined with newlines. */
  text: string;
  images: CanonImage[];
  toolCalls: CanonToolCall[];
  toolResults: CanonToolResult[];
}

export interface CanonTool {
  name: string;
  description: string;
  schema: Json;
}

export interface CanonRequest {
  /** The system text, joined as the parsers join it; null when the request carries none. */
  system: string | null;
  turns: CanonTurn[];
  tools: CanonTool[];
  toolChoice: { kind: 'auto' | 'any' | 'none' | 'tool'; name?: string } | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  /** The client asked for JSON mode (an OpenAI response_format; Anthropic bodies cannot say so). */
  jsonMode: boolean;
  stream: boolean;
}

export interface CanonAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const str = (v: unknown): string => (typeof v === 'string' ? v : '');

/** A content value's text as every wire's parser reads it: a string itself, else its text blocks joined. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return v === '' ? [] : [v];
  if (!Array.isArray(v)) return [];
  return v.flatMap((x) => (typeof x === 'string' ? [x] : []));
}

function argsObject(args: string): Json {
  try {
    const parsed: unknown = JSON.parse(args);
    if (isRecord(parsed)) return parsed;
  } catch {
    // Not JSON: an empty input object.
  }
  return {};
}

/** The data URL of an image, split into its media type and base64 payload; null when it is not one. */
function splitDataUrl(url: string): { mediaType: string; data: string } | null {
  const match = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (match === null || match[2] === undefined) return null;
  return { mediaType: match[1] === '' ? 'image/png' : match[1], data: match[3] };
}

// ---------------------------------------------------------------------------
/// Canonicalising the two client wires.
// ---------------------------------------------------------------------------

/** An OpenAI Chat Completions body as the canonical request. */
export function canonFromOpenAI(body: Json): CanonRequest {
  const systemParts: string[] = [];
  const turns: CanonTurn[] = [];
  const callNames = new Map<string, string>();
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      turns.push({
        role: 'user',
        text: '',
        images: [],
        toolCalls: [],
        toolResults: [{ id: str(raw.tool_call_id), name: '', text: typeof raw.content === 'string' ? raw.content : blocksText(raw.content) }],
      });
      continue;
    }
    const turn: CanonTurn = {
      role: raw.role === 'assistant' ? 'assistant' : 'user',
      text: blocksText(raw.content),
      images: [],
      toolCalls: [],
      toolResults: [],
    };
    if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part) || part.type !== 'image_url' || !isRecord(part.image_url)) continue;
        const url = str(part.image_url.url);
        const data = splitDataUrl(url);
        turn.images.push(data !== null ? { mediaType: data.mediaType, data: data.data, url: null } : { mediaType: '', data: null, url });
      }
    }
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const entry: CanonToolCall = { id: str(call.id), name: str(call.function.name), args: str(call.function.arguments) || '{}' };
        turn.toolCalls.push(entry);
        if (entry.id !== '') callNames.set(entry.id, entry.name);
      }
    }
    turns.push(turn);
  }
  for (const turn of turns) for (const result of turn.toolResults) result.name = callNames.get(result.id) ?? '';
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: str(raw.function.description),
        schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CanonRequest['toolChoice'] = null;
  if (body.tool_choice === 'auto') toolChoice = { kind: 'auto' };
  else if (body.tool_choice === 'none') toolChoice = { kind: 'none' };
  else if (body.tool_choice === 'required') toolChoice = { kind: 'any' };
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { kind: 'tool', name: body.tool_choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.length > 0 ? systemParts.join('\n') : null,
    turns,
    tools,
    toolChoice,
    maxTokens: num(body.max_tokens) ?? num(body.max_completion_tokens),
    temperature: num(body.temperature),
    topP: num(body.top_p),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** A tool_result block's content as text: its text blocks, else its JSON. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An Anthropic Messages body as the canonical request. */
export function canonFromAnthropic(body: Json): CanonRequest {
  const systemText = blocksText(body.system);
  const turns: CanonTurn[] = [];
  const callNames = new Map<string, string>();
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const turn: CanonTurn = {
      role: raw.role === 'assistant' ? 'assistant' : 'user',
      text: blocksText(raw.content),
      images: [],
      toolCalls: [],
      toolResults: [],
    };
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            turn.images.push({ mediaType: str(block.source.media_type) || 'image/png', data: block.source.data, url: null });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            turn.images.push({ mediaType: '', data: null, url: block.source.url });
          }
        } else if (block.type === 'tool_use') {
          const entry: CanonToolCall = {
            id: str(block.id),
            name: str(block.name),
            args: JSON.stringify(isRecord(block.input) ? block.input : {}),
          };
          turn.toolCalls.push(entry);
          if (entry.id !== '') callNames.set(entry.id, entry.name);
        } else if (block.type === 'tool_result') {
          turn.toolResults.push({ id: str(block.tool_use_id), name: '', text: toolResultText(block.content) });
        }
      }
    }
    turns.push(turn);
  }
  for (const turn of turns) for (const result of turn.toolResults) result.name = callNames.get(result.id) ?? '';
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: str(raw.description),
        schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CanonRequest['toolChoice'] = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = { kind: 'auto' };
    else if (body.tool_choice.type === 'any') toolChoice = { kind: 'any' };
    else if (body.tool_choice.type === 'none') toolChoice = { kind: 'none' };
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { kind: 'tool', name: body.tool_choice.name };
  }
  return {
    system: systemText !== '' ? systemText : null,
    turns,
    tools,
    toolChoice,
    maxTokens: num(body.max_tokens),
    temperature: num(body.temperature),
    topP: num(body.top_p),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
/// Rendering the canonical request into each provider wire.
// ---------------------------------------------------------------------------

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
export function renderOpenAI(c: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (c.system !== null) messages.push({ role: 'system', content: c.system });
  for (const turn of c.turns) {
    for (const result of turn.toolResults) {
      messages.push({ role: 'tool', tool_call_id: result.id, content: result.text });
    }
    if (turn.role === 'assistant') {
      messages.push({
        role: 'assistant',
        content: turn.text === '' && turn.toolCalls.length > 0 ? null : turn.text,
        ...(turn.toolCalls.length > 0
          ? { tool_calls: turn.toolCalls.map((call) => ({ id: call.id, type: 'function', function: { name: call.name, arguments: call.args } })) }
          : {}),
      });
      continue;
    }
    if (turn.images.length === 0 && turn.toolCalls.length === 0) {
      if (turn.text !== '' || turn.toolResults.length === 0) messages.push({ role: 'user', content: turn.text });
      continue;
    }
    const parts: Json[] = [];
    if (turn.text !== '') parts.push({ type: 'text', text: turn.text });
    for (const image of turn.images) {
      const url = image.data !== null ? `data:${image.mediaType};base64,${image.data}` : (image.url ?? '');
      parts.push({ type: 'image_url', image_url: { url } });
    }
    messages.push({ role: 'user', content: parts });
  }
  // A user turn of nothing but tool results leaves no user message behind it; the OpenAI parser
  // reads the last user message's text, so close with an empty one to keep that text empty.
  const lastTurn = c.turns[c.turns.length - 1];
  const lastMessage = messages[messages.length - 1];
  if (lastTurn !== undefined && lastTurn.role === 'user' && lastMessage !== undefined && lastMessage.role === 'tool') {
    messages.push({ role: 'user', content: '' });
  }
  const out: Json = { model, messages };
  if (c.maxTokens !== null) out.max_tokens = c.maxTokens;
  if (c.temperature !== null) out.temperature = c.temperature;
  if (c.topP !== null) out.top_p = c.topP;
  if (c.stops.length > 0) out.stop = c.stops;
  if (c.tools.length > 0) {
    out.tools = c.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
  }
  if (c.toolChoice !== null) {
    if (c.toolChoice.kind === 'auto') out.tool_choice = 'auto';
    else if (c.toolChoice.kind === 'none') out.tool_choice = 'none';
    else if (c.toolChoice.kind === 'any') out.tool_choice = 'required';
    else out.tool_choice = { type: 'function', function: { name: c.toolChoice.name ?? '' } };
  }
  if (c.jsonMode) out.response_format = { type: 'json_object' };
  if (c.stream) {
    out.stream = true;
    // The client's usage check reads a usage chunk off the stream; ask the provider for one.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The canonical request as an Anthropic Messages body naming `model`. */
export function renderAnthropic(c: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const turn of c.turns) {
    const blocks: Json[] = [];
    for (const result of turn.toolResults) blocks.push({ type: 'tool_result', tool_use_id: result.id, content: result.text });
    if (turn.text !== '') blocks.push({ type: 'text', text: turn.text });
    for (const image of turn.images) {
      blocks.push(
        image.data !== null
          ? { type: 'image', source: { type: 'base64', media_type: image.mediaType || 'image/png', data: image.data } }
          : { type: 'image', source: { type: 'url', url: image.url ?? '' } },
      );
    }
    for (const call of turn.toolCalls) {
      blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: argsObject(call.args) });
    }
    if (blocks.length === 0) messages.push({ role: turn.role, content: '' });
    else if (blocks.length === 1 && blocks[0].type === 'text') messages.push({ role: turn.role, content: turn.text });
    else messages.push({ role: turn.role, content: blocks });
  }
  const out: Json = { model, messages };
  if (c.maxTokens !== null) out.max_tokens = c.maxTokens;
  if (c.system !== null) out.system = c.system;
  if (c.temperature !== null) out.temperature = c.temperature;
  if (c.topP !== null) out.top_p = c.topP;
  if (c.stops.length > 0) out.stop_sequences = c.stops;
  if (c.tools.length > 0) {
    out.tools = c.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (c.toolChoice !== null) {
    if (c.toolChoice.kind === 'auto') out.tool_choice = { type: 'auto' };
    else if (c.toolChoice.kind === 'any') out.tool_choice = { type: 'any' };
    else if (c.toolChoice.kind === 'none') out.tool_choice = { type: 'none' };
    else out.tool_choice = { type: 'tool', name: c.toolChoice.name ?? '' };
  }
  if (c.stream) out.stream = true;
  return out;
}

/** The canonical request as a Gemini generateContent body for `model` (the model rides in the URL). */
export function renderGemini(c: CanonRequest): Json {
  const contents: Json[] = [];
  for (const turn of c.turns) {
    const parts: Json[] = [];
    for (const result of turn.toolResults) parts.push({ functionResponse: { name: result.name, response: { result: result.text } } });
    if (turn.text !== '') parts.push({ text: turn.text });
    for (const image of turn.images) {
      parts.push(
        image.data !== null
          ? { inlineData: { mimeType: image.mediaType || 'image/png', data: image.data } }
          : { fileData: { fileUri: image.url ?? '' } },
      );
    }
    for (const call of turn.toolCalls) parts.push({ functionCall: { name: call.name, args: argsObject(call.args) } });
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: turn.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (c.system !== null) out.systemInstruction = { parts: [{ text: c.system }] };
  if (c.tools.length > 0) {
    out.tools = [{ functionDeclarations: c.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  }
  if (c.toolChoice !== null) {
    const config: Json =
      c.toolChoice.kind === 'auto'
        ? { mode: 'AUTO' }
        : c.toolChoice.kind === 'none'
          ? { mode: 'NONE' }
          : c.toolChoice.kind === 'any'
            ? { mode: 'ANY' }
            : { mode: 'ANY', allowedFunctionNames: [c.toolChoice.name ?? ''] };
    out.toolConfig = { functionCallingConfig: config };
  }
  const generation: Json = {};
  if (c.maxTokens !== null) generation.maxOutputTokens = c.maxTokens;
  if (c.temperature !== null) generation.temperature = c.temperature;
  if (c.topP !== null) generation.topP = c.topP;
  if (c.stops.length > 0) generation.stopSequences = c.stops;
  if (c.jsonMode) generation.responseMimeType = 'application/json';
  if (Object.keys(generation).length > 0) out.generationConfig = generation;
  return out;
}

// ---------------------------------------------------------------------------
/// Reading a provider's answer, in whichever wire, into the canonical answer.
// ---------------------------------------------------------------------------

function usagePair(input: unknown, output: unknown): { inputTokens: number; outputTokens: number } {
  return { inputTokens: num(input) ?? 0, outputTokens: num(output) ?? 0 };
}

/** An OpenAI-shaped answer body (a JSON completion or its SSE chunks) as the canonical answer. */
export function answerFromOpenAI(body: unknown, sseFrames: string[] | null): CanonAnswer {
  if (sseFrames === null) {
    if (!isRecord(body)) return { text: '', ...usagePair(0, 0) };
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const message = isRecord(choices[0]) && isRecord((choices[0] as Json).message) ? ((choices[0] as Json).message as Json) : {};
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text: blocksText(message.content), ...usagePair(usage.prompt_tokens, usage.completion_tokens) };
  }
  let text = '';
  let usage = usagePair(0, 0);
  for (const frame of sseFrames) {
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) usage = usagePair(chunk.usage.prompt_tokens, chunk.usage.completion_tokens);
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const delta = isRecord(choices[0]) && isRecord((choices[0] as Json).delta) ? ((choices[0] as Json).delta as Json) : null;
    if (delta !== null && typeof delta.content === 'string') text += delta.content;
  }
  return { text, ...usage };
}

/** An Anthropic-shaped answer body (a JSON message or its named SSE events) as the canonical answer. */
export function answerFromAnthropic(body: unknown, sseEvents: { event: string; data: string }[] | null): CanonAnswer {
  if (sseEvents === null) {
    if (!isRecord(body)) return { text: '', ...usagePair(0, 0) };
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text: blocksText(body.content), ...usagePair(usage.input_tokens, usage.output_tokens) };
  }
  let text = '';
  let input = 0;
  let output = 0;
  for (const { event, data } of sseEvents) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(data);
    } catch {
      continue;
    }
    if (!isRecord(parsed)) continue;
    if (event === 'message_start' && isRecord(parsed.message) && isRecord(parsed.message.usage)) {
      input = num(parsed.message.usage.input_tokens) ?? input;
    } else if (event === 'content_block_delta' && isRecord(parsed.delta) && parsed.delta.type === 'text_delta') {
      text += str(parsed.delta.text);
    } else if (event === 'message_delta' && isRecord(parsed.usage)) {
      output = num(parsed.usage.output_tokens) ?? output;
    }
  }
  return { text, inputTokens: input, outputTokens: output };
}

/** A Gemini-shaped answer body (a JSON response or its SSE chunks) as the canonical answer. */
export function answerFromGemini(body: unknown, sseFrames: string[] | null): CanonAnswer {
  const fromResponse = (response: Json, into: CanonAnswer): void => {
    const candidates = Array.isArray(response.candidates) ? response.candidates : [];
    const content = isRecord(candidates[0]) && isRecord((candidates[0] as Json).content) ? ((candidates[0] as Json).content as Json) : null;
    if (content !== null && Array.isArray(content.parts)) {
      for (const part of content.parts) {
        if (isRecord(part) && typeof part.text === 'string') into.text += part.text;
      }
    }
    if (isRecord(response.usageMetadata)) {
      into.inputTokens = num(response.usageMetadata.promptTokenCount) ?? into.inputTokens;
      into.outputTokens = num(response.usageMetadata.candidatesTokenCount) ?? into.outputTokens;
    }
  };
  const answer: CanonAnswer = { text: '', inputTokens: 0, outputTokens: 0 };
  if (sseFrames === null) {
    if (isRecord(body)) fromResponse(body, answer);
    return answer;
  }
  for (const frame of sseFrames) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(frame);
    } catch {
      continue;
    }
    if (isRecord(parsed)) fromResponse(parsed, answer);
  }
  return answer;
}

// ---------------------------------------------------------------------------
/// Rendering the canonical answer in the client's wire.
// ---------------------------------------------------------------------------

/** A fixed epoch second for completion documents: nothing in a reply reads the clock. */
export const FIXED_CREATED = 1767225600;

let idCounter = 0;
/** A deterministic answer id: a per-process counter, never the clock. */
export function nextId(prefix: string): string {
  idCounter += 1;
  return `${prefix}_${idCounter}`;
}

/** Split a text into stream pieces of at most `size` characters (the concatenation is the text).
 * The size only shapes the client's reading rhythm; no check counts the frames. */
export function splitPieces(text: string, size = 32): string[] {
  const pieces: string[] = [];
  for (let i = 0; i < text.length; i += size) pieces.push(text.slice(i, i + size));
  return pieces;
}

/** The canonical answer as an OpenAI Chat Completions body. */
export function renderOpenAIAnswer(a: CanonAnswer, model: string): Json {
  return {
    id: nextId('chatcmpl'),
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: a.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: a.inputTokens, completion_tokens: a.outputTokens, total_tokens: a.inputTokens + a.outputTokens },
  };
}

/** The canonical answer as an Anthropic Messages body; `model` is the name the client used. */
export function renderAnthropicAnswer(a: CanonAnswer, model: string): Json {
  return {
    id: nextId('msg'),
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: a.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: a.inputTokens, output_tokens: a.outputTokens },
  };
}

const sse = (event: string | null, data: unknown): string =>
  `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;

/** The canonical answer as an OpenAI SSE stream: role chunk, text chunks, finish, usage, [DONE]. */
export function renderOpenAIStream(a: CanonAnswer, model: string): string {
  const id = nextId('chatcmpl');
  const chunk = (delta: Json, finish: string | null): Json => ({
    id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sse(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitPieces(a.text)) frames.push(sse(null, chunk({ content: piece }, null)));
  frames.push(sse(null, chunk({}, 'stop')));
  frames.push(
    sse(null, {
      id,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: a.inputTokens, completion_tokens: a.outputTokens, total_tokens: a.inputTokens + a.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** The canonical answer as an Anthropic SSE stream, in the Messages event order. */
export function renderAnthropicStream(a: CanonAnswer, model: string): string {
  const frames: string[] = [
    sse('message_start', {
      type: 'message_start',
      message: {
        id: nextId('msg'),
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: a.inputTokens, output_tokens: 0 },
      },
    }),
    sse('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitPieces(a.text)) {
    frames.push(sse('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sse('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sse('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: a.outputTokens },
    }),
    sse('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** Split a raw SSE body into its frames: data payloads, and named events with their payloads. */
export function parseSse(raw: string): { frames: string[]; events: { event: string; data: string }[] } {
  const frames: string[] = [];
  const events: { event: string; data: string }[] = [];
  for (const block of raw.replace(/\r\n/g, '\n').split('\n\n')) {
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length === 0) continue;
    const data = dataLines.join('\n');
    if (data === '[DONE]') continue;
    frames.push(data);
    if (event !== null) events.push({ event, data });
  }
  return { frames, events };
}

/** A provider refusal's cause, read from its error message: spent budgets and outages never heal. */
export function refusalCause(status: number, bodyText: string): 'budget' | 'outage' | 'transient' {
  let message: string | null = null;
  try {
    const parsed: unknown = JSON.parse(bodyText);
    if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.message === 'string') message = parsed.error.message;
  } catch {
    // Not JSON: fall through to the status.
  }
  // The wire's error message names the cause exactly ("...'s free-tier budget is spent",
  // "... is out", "... refused this call"); trust it over the status, because the one provider
  // that refuses with 402 uses that status for its scheduled refusals too.
  if (message !== null) {
    if (/budget is spent/i.test(message)) return 'budget';
    if (/\bis out\b/i.test(message)) return 'outage';
    return 'transient';
  }
  return status === 402 ? 'budget' : 'transient';
}
