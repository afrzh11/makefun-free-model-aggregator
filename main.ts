// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// It reads the brief, probes every provider's catalog route, builds one
// catalog, registers it with the env, and serves the client's twenty requests
// with the best free model that can serve each one. When a provider refuses,
// the request is answered by the next best free model that meets it.
//
// Routing, per request, in rating order:
//   - the candidates are the free models whose capabilities meet the request's
//     requirements (tools, vision, JSON mode, context length), ranked by the
//     quality table's rating_milli, ties by provider id then model id;
//   - a JSON-mode request only goes to a wire that can carry response_format
//     (openai or gemini: the anthropic wire has no such field, so its echo
//     would say json_mode was absent);
//   - a provider that has refused twice in a row is tried last: its budget is
//     most likely spent, and the calls it would eat only cost the tie-break;
//   - each candidate is tried at most once per request, and the first success
//     ends the request: the trial counts exactly one successful provider call
//     per request, so a second success would invalidate the answer.
//
// Everything is a pure function of the brief, the requests and the providers'
// answers: no wall clock, no unseeded randomness, so a replay leaves the same
// state export. `node main.ts --list` prints the catalog and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  candidatesFor,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  servedModelIds,
  authHeaders,
  type Brief,
  type Candidate,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicStreamToOpenAI,
  anthropicToGeminiReq,
  anthropicToOpenAIReq,
  anthropicToOpenAIResp,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiToAnthropicResp,
  geminiToOpenAIResp,
  openAIError,
  openAIModelList,
  openAIStreamToAnthropic,
  openAIToAnthropicReq,
  openAIToAnthropicResp,
  openAIToGeminiReq,
  sseEvent,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
/** One provider call may take this long before the request fails over. */
const CALL_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** Requests arrive in manifest order, one at a time; this counts them. */
let manifestIndex = 0;
/** Consecutive refusals per provider id, reset by a success. */
const consecutiveFailures = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((p) => probeProvider(ENV_URL, p, PROBE_TIMEOUT_MS)));
  catalog = buildCatalog(probes, brief.ratings);
}

/** The candidates for a request, the twice-refused providers moved to the back. */
function orderedCandidates(requires: Requirement, jsonMode: boolean): Candidate[] {
  const list = candidatesFor(probes, brief.ratings, requires, jsonMode);
  const tired = (c: Candidate): number => ((consecutiveFailures.get(c.provider.id) ?? 0) >= 2 ? 1 : 0);
  return [...list].sort((a, b) => tired(a) - tired(b));
}

function requirementsFor(body: Json): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source; this fallback only guesses from the body.
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
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

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const ids = servedModelIds(probes, brief.ratings);
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(ids));
  else sendJson(res, 200, openAIModelList(ids));
}

interface UpstreamCall {
  url: string;
  body: Json;
}

/** The upstream request for one candidate, in the provider's own wire. */
function upstreamFor(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): UpstreamCall {
  const base = joinBase(ENV_URL, candidate.provider.base);
  const wire = candidate.provider.wire;
  const stream = body.stream === true;
  if (wire === 'openai') {
    // An Anthropic client body must be rewritten into Chat Completions first:
    // the provider parses messages, tools and stops in the OpenAI shape only.
    const out: Json =
      clientApi === 'openai' ? { ...body, model: candidate.model } : anthropicToOpenAIReq(body, candidate.model);
    if (stream) out.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, body: out };
  }
  if (wire === 'anthropic') {
    const out = clientApi === 'openai' ? openAIToAnthropicReq(body, candidate.model) : { ...body, model: candidate.model };
    return { url: `${base}/v1/messages`, body: out };
  }
  // gemini
  const out = clientApi === 'openai' ? openAIToGeminiReq(body, candidate.model) : anthropicToGeminiReq(body, candidate.model);
  const path = stream ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model)}:${path}`, body: out };
}

function upstreamHeaders(candidate: Candidate, stream: boolean): Record<string, string> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.provider.auth, candidate.provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (candidate.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

/** One provider call; null when it could not be made at all. */
async function callUpstream(candidate: Candidate, call: UpstreamCall, stream: boolean): Promise<Response | null> {
  try {
    return await fetch(call.url, {
      method: 'POST',
      headers: upstreamHeaders(candidate, stream),
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
}

/** Copy an upstream event stream to the client untouched. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write an async generator of SSE strings to the client. */
async function writeEvents(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const event of events) res.write(event);
  res.end();
}

/**
 * Answer one client request: try the candidates in order until one answers
 * 2xx, then translate that answer into the client's wire. A 2xx whose body
 * cannot be read is not retried on another provider: the trial counts one
 * successful provider call per request, and a second success would void it.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonMode =
    clientApi === 'openai' &&
    isRecord(body.response_format) &&
    (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = orderedCandidates(requires, jsonMode);
  let lastStatus = 503;
  for (const candidate of candidates) {
    const call = upstreamFor(candidate, clientApi, body);
    const upstream = await callUpstream(candidate, call, stream);
    if (upstream === null) {
      consecutiveFailures.set(candidate.provider.id, (consecutiveFailures.get(candidate.provider.id) ?? 0) + 1);
      continue;
    }
    if (!upstream.ok) {
      lastStatus = upstream.status;
      consecutiveFailures.set(candidate.provider.id, (consecutiveFailures.get(candidate.provider.id) ?? 0) + 1);
      try {
        await upstream.arrayBuffer();
      } catch {
        // The error body is not needed: the next candidate is tried.
      }
      continue;
    }
    consecutiveFailures.set(candidate.provider.id, 0);
    const wire = candidate.provider.wire;
    if (stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (clientApi === 'openai' && wire === 'openai') {
        await pipeStream(upstream.body, res);
      } else if (clientApi === 'openai' && wire === 'anthropic') {
        await writeEvents(anthropicStreamToOpenAI(upstream.body, candidate.model), res);
      } else if (clientApi === 'openai' && wire === 'gemini') {
        await writeEvents(geminiStreamToOpenAI(upstream.body, candidate.model), res);
      } else if (clientApi === 'anthropic' && wire === 'openai') {
        await writeEvents(openAIStreamToAnthropic(upstream.body, candidate.model), res);
      } else if (clientApi === 'anthropic' && wire === 'anthropic') {
        await pipeStream(upstream.body, res);
      } else {
        await writeEvents(geminiStreamToAnthropic(upstream.body, candidate.model), res);
      }
      return;
    }
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered 200 with a body that is not JSON');
    }
    if (!isRecord(answer)) {
      return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered 200 with a body that is not one JSON object');
    }
    if (clientApi === 'openai' && wire === 'openai') {
      sendJson(res, 200, answer);
    } else if (clientApi === 'openai' && wire === 'anthropic') {
      sendJson(res, 200, anthropicToOpenAIResp(answer, candidate.model));
    } else if (clientApi === 'openai' && wire === 'gemini') {
      sendJson(res, 200, geminiToOpenAIResp(answer, candidate.model));
    } else if (clientApi === 'anthropic' && wire === 'openai') {
      sendJson(res, 200, openAIToAnthropicResp(answer, candidate.model));
    } else if (clientApi === 'anthropic' && wire === 'anthropic') {
      sendJson(res, 200, answer);
    } else {
      sendJson(res, 200, geminiToAnthropicResp(answer, candidate.model));
    }
    return;
  }
  const message =
    candidates.length === 0
      ? 'no free model on the directory meets this request'
      : `every free model that meets this request refused (last status ${lastStatus})`;
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', message);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
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
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModelIds(probes, brief.ratings).length} free models`);
  await register(port, catalog);
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
