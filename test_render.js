const assert = require("assert");
const MD = require("./render.js");

const table = MD.render("| Field |\n| :--- |\n| wbsnodeid [1] |\n| scope [2] |");
assert(table.includes("<table>"), "table not rendered");
assert(table.includes("<th>Field</th>"), "table header missing");
assert(table.includes("<td>wbsnodeid"), "table cell missing");
assert(!/\|/.test(table), "raw pipes left in table output");

MD.setSources(2);
const prose = MD.render("**Bold** and `code`\n\n# Heading\n\n- one\n- two\n\nSee [1, 3] for detail.");
assert(prose.includes("<strong>Bold</strong>"), "bold not rendered");
assert(prose.includes("<code>code</code>"), "code not rendered");
assert(/<h\d/.test(prose), "heading not rendered");
assert(prose.includes("<li>one</li>"), "list not rendered");
assert(prose.includes('data-n="1"'), "in-range citation not linked");
assert(!/data-n="3"/.test(prose), "out-of-range citation should stay plain");

MD.setSources(0);
const doc = MD.render("See [1] and [2].");
assert(!/class="cite"/.test(doc), "no citation context should leave [n] plain");

console.log("render tests passed");
