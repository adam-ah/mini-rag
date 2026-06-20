const SSE = (function () {
  function frameData(frame) {
    const data = [];
    for (const line of frame.split(/\r?\n/)) {
      if (!line.startsWith("data:")) continue;
      let value = line.slice(5);
      if (value.startsWith(" ")) value = value.slice(1);
      data.push(value);
    }
    return data.length ? data.join("\n") : null;
  }

  function consume(buffer, flush) {
    const payloads = [];
    const boundary = /\r?\n\r?\n/g;
    let start = 0;
    let match;
    while ((match = boundary.exec(buffer)) !== null) {
      const payload = frameData(buffer.slice(start, match.index));
      if (payload !== null) payloads.push(payload);
      start = boundary.lastIndex;
    }

    let rest = buffer.slice(start);
    if (flush && rest.trim()) {
      const payload = frameData(rest);
      if (payload !== null) payloads.push(payload);
      rest = "";
    }
    return { payloads, rest };
  }

  return { consume };
})();

if (typeof module !== "undefined") module.exports = SSE;
