const assert = require("assert");
const SSE = require("./stream.js");

function consumeChunks(chunks, flush = true) {
  let buffer = "";
  const payloads = [];
  for (const chunk of chunks) {
    buffer += chunk;
    const parsed = SSE.consume(buffer, false);
    buffer = parsed.rest;
    payloads.push(...parsed.payloads);
  }
  const parsed = SSE.consume(buffer, flush);
  payloads.push(...parsed.payloads);
  return { payloads, rest: parsed.rest };
}

const stream =
  'data: {"type":"step","label":"one"}\n\n' +
  'data: {"type":"token","text":"a\\n\\nb"}\n\n' +
  'data: {"type":"done"}\n\n';
const chunks = [stream.slice(0, 7), stream.slice(7, 45), stream.slice(45, 68), stream.slice(68)];
const parsed = consumeChunks(chunks);
assert.deepStrictEqual(parsed.payloads.map(JSON.parse).map(x => x.type), ["step", "token", "done"]);
assert.strictEqual(JSON.parse(parsed.payloads[1]).text, "a\n\nb");
assert.strictEqual(parsed.rest, "");

const crlf = consumeChunks(['data: {"type":"step"}\r\n\r\ndata: {"type":"done"}\r\n\r\n']);
assert.deepStrictEqual(crlf.payloads.map(JSON.parse).map(x => x.type), ["step", "done"]);

const partial = SSE.consume('data: {"type":"step"}', false);
assert.deepStrictEqual(partial.payloads, []);
assert.notStrictEqual(partial.rest, "");
assert.deepStrictEqual(SSE.consume(partial.rest, true).payloads.map(JSON.parse).map(x => x.type), ["step"]);

console.log("stream parser tests passed");
