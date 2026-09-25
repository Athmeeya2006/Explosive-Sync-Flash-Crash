// Build-time TeX -> SVG renderer for research/index.html.
// stdin : JSON array of {"tex": "...", "display": true|false}
// stdout: JSON array of {"svg": "<mjx-container ...>...</mjx-container>"} or {"error": "..."}
// Any TeX error is reported per item (never silently rendered as red text).

const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

const packages = AllPackages.filter((p) => !["autoload", "require", "noundefined", "noerrors"].includes(p));
const tex = new TeX({
  packages,
  formatError: (jax, err) => {
    throw err;
  },
});
const svg = new SVG({ fontCache: "local" });
const doc = mathjax.document("", { InputJax: tex, OutputJax: svg });

let input = "";
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  const items = JSON.parse(input);
  const out = items.map(({ tex: src, display }) => {
    try {
      const node = doc.convert(src, { display, em: 16, ex: 8, containerWidth: 1100 });
      return { svg: adaptor.outerHTML(node) };
    } catch (e) {
      return { error: String(e && e.message ? e.message : e) };
    }
  });
  process.stdout.write(JSON.stringify(out));
});
