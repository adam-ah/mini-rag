const MD = (function () {
  let srcN = 0;

  function setSources(n) { srcN = n || 0; }

  function esc(s) {
    return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }

  function linkCites(html) {
    return html.replace(/\[([\d\s,;&–-]+)\]/g, (full, inner) => {
      if (!/\d/.test(inner)) return full;
      const linked = inner.replace(/\d+/g, d => {
        const n = parseInt(d, 10);
        if (!srcN || n < 1 || n > srcN) return d;
        return `<a class="cite" data-n="${d}">${d}</a>`;
      });
      return "[" + linked + "]";
    });
  }

  function inline(s) {
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^_])_([^_]+)_/g, "$1<em>$2</em>");
    return linkCites(s);
  }

  function tableCells(r) { return r.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map(c => c.trim()); }
  function isTableSep(r) { return /^[\s|:-]+$/.test(r) && r.includes("-"); }

  function renderTable(rows) {
    const hasHead = rows.length >= 2 && isTableSep(rows[1]);
    let h = "<table>";
    if (hasHead) h += "<thead><tr>" + tableCells(rows[0]).map(c => `<th>${inline(c)}</th>`).join("") + "</tr></thead>";
    h += "<tbody>";
    for (let i = hasHead ? 2 : 0; i < rows.length; i++) h += "<tr>" + tableCells(rows[i]).map(c => `<td>${inline(c)}</td>`).join("") + "</tr>";
    return h + "</tbody></table>";
  }

  function render(t) {
    const lines = esc(t).split("\n");
    let html = "", para = [], listOpen = false, i = 0;
    const flushPara = () => { if (para.length) { html += "<p>" + para.map(inline).join("<br>") + "</p>"; para = []; } };
    const closeList = () => { if (listOpen) { html += "</ul>"; listOpen = false; } };
    while (i < lines.length) {
      const line = lines[i].replace(/\s+$/, "");
      let m;
      if (/^\s*$/.test(line)) { flushPara(); closeList(); i++; continue; }
      if ((m = line.match(/^(#{1,4})\s+(.*)$/))) {
        flushPara(); closeList();
        const lv = Math.min(6, m[1].length + 2);
        html += `<h${lv}>` + inline(m[2]) + `</h${lv}>`; i++; continue;
      }
      if (line.trim().startsWith("|")) {
        let j = i; const rows = [];
        while (j < lines.length && lines[j].trim().startsWith("|")) { rows.push(lines[j].trim()); j++; }
        if (rows.length >= 2 && isTableSep(rows[1])) { flushPara(); closeList(); html += renderTable(rows); i = j; continue; }
      }
      if ((m = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/))) {
        flushPara();
        if (!listOpen) { html += "<ul>"; listOpen = true; }
        html += "<li>" + inline(m[1]) + "</li>"; i++; continue;
      }
      para.push(line); i++;
    }
    flushPara(); closeList();
    return html;
  }

  return { render, setSources, esc };
})();

if (typeof module !== "undefined") module.exports = MD;
