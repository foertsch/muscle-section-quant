// Report base: DOM and number helpers, the segmented control, the tooltip, the enlarged-image dialog, percentiles,
// rows of histograms, pan and zoom, and the display choices kept between visits. Page logic stays in the page.
const $ = q => document.querySelector(q);
const fmt = (v, d = 0) => (v == null || !isFinite(v)) ? "n/a"
  : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

// One button per [value, label]; the pressed one is state[key]; onChange runs after a click.
function seg(host, options, state, key, onChange) {
  host.innerHTML = "";
  for (const [val, label] of options) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label; b.dataset.val = val;
    b.setAttribute("aria-pressed", String(state[key] === val));
    b.addEventListener("click", () => {
      state[key] = val;
      for (const x of host.children) x.setAttribute("aria-pressed", String(x === b));
      onChange();
    });
    host.appendChild(b);
  }
}

const tip = (() => {
  const el = document.body.appendChild(Object.assign(document.createElement("div"), { id: "tip" }));
  const move = e => {
    const pad = 14, w = el.offsetWidth, h = el.offsetHeight;
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > innerWidth - 8) x = e.clientX - w - pad;
    if (y + h > innerHeight - 8) y = e.clientY - h - pad;
    el.style.left = Math.max(8, x) + "px"; el.style.top = Math.max(8, y) + "px";
  };
  return { el, move, show: (html, e) => { el.innerHTML = html; el.style.display = "block"; move(e); },
           hide: () => { el.style.display = "none"; } };
})();

// The node in a modal dialog with a title and a Close button.
const lightbox = (() => {
  let dlg;
  return (title, node) => {
    if (!dlg) {
      dlg = document.body.appendChild(Object.assign(document.createElement("dialog"), {
        innerHTML: `<div class="zhead"><span class="ztitle"></span><button class="zclose" type="button">Close</button></div><div class="zbody"></div>` }));
      dlg.setAttribute("aria-label", "Enlarged image");
      dlg.addEventListener("click", e => { if (e.target === dlg || e.target.closest(".zclose")) dlg.close(); });
      dlg.addEventListener("close", () => { tip.hide(); document.body.appendChild(tip.el); });
    }
    dlg.querySelector(".zbody").replaceChildren(node);
    dlg.querySelector(".ztitle").textContent = title;
    tip.hide();
    dlg.appendChild(tip.el);   // a modal dialog sits above the page, so the tooltip has to live inside it
    dlg.showModal();
  };
})();

// numpy's default (linear) percentile of an ascending array, including its interpolation order
function pctl(a, p) {
  if (!a.length) return NaN;
  const h = (a.length - 1) * (p / 100), lo = Math.floor(h), t = h - lo;
  if (lo + 1 >= a.length) return a[a.length - 1];
  const x = a[lo], y = a[lo + 1], d = y - x;
  return t >= 0.5 ? y - d * (1 - t) : x + d * t;
}
function countGE(a, b) {   // how many of the ascending array a are at least b
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < b) lo = mid + 1; else hi = mid; }
  return a.length - lo;
}
function niceStep(span, target) {
  const raw = span / target, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2.5 : m < 7.5 ? 5 : 10) * p;
}

// One row of bars per entry of cfg.rows on a shared x-axis and y-scale; hovering a bin lists every row's value.
//   cfg.edges   bin edges; cfg.dec  decimals shown in the tooltip; cfg.aria  chart description
//   row: { name: [strong, rest], sub, color, a: % per bin, b: second series drawn as a step line (clipped to the
//          row), marker: x of a line across the row, stat: [text, text] to the right }
//   cfg.vline   { x, label } dashed line across all rows; cfg.split  x: bars below it are drawn faded
//   cfg.strip   [lo, hi] draws the grey ramp of an image display range under the axis
//   cfg.onScale(maxY) is told the tallest bar
function rowHist(host, cfg) {
  const rows = cfg.rows, W = Math.max(320, host.clientWidth || 900), narrow = W < 640;
  const L = narrow ? 86 : 150, R = narrow ? 104 : 150, T = cfg.vline ? 26 : 10;
  const rowH = narrow ? 42 : 52, gap = 8;
  const plotBottom = T + rows.length * (rowH + gap) - gap, H = plotBottom + (cfg.strip ? 42 : 24);
  const e = cfg.edges, x0 = e[0], x1 = e[e.length - 1], bw = e[1] - e[0];
  const X = v => L + (Math.min(Math.max(v, x0), x1) - x0) / (x1 - x0) * (W - L - R);
  let maxY = 0;
  for (const r of rows) for (const v of r.a) maxY = Math.max(maxY, v);
  const Y = p => Math.min(1, p / maxY);
  const xs = cfg.split == null ? null : X(cfg.split);
  let g = "";
  if (cfg.strip) {
    const [lo, hi] = cfg.strip;
    g += `<defs><linearGradient id="grad-${host.id}" gradientUnits="userSpaceOnUse" x1="${X(lo)}" x2="${Math.max(X(hi), X(lo) + 1)}" y1="0" y2="0">`
      + `<stop offset="0" stop-color="#000"/><stop offset="1" stop-color="#fff"/></linearGradient></defs>`;
  }
  const step = niceStep(x1 - x0, narrow ? 4 : 9);
  for (let t = Math.ceil(x0 / step) * step; t <= x1 + 1e-9; t += step) {
    const x = X(t).toFixed(1);
    g += `<line class="grid" x1="${x}" x2="${x}" y1="${T}" y2="${plotBottom}"/>`
      + `<text x="${x}" y="${plotBottom + 14}" text-anchor="middle">${fmt(t)}</text>`;
  }
  rows.forEach((r, i) => {
    const top = T + i * (rowH + gap), base = top + rowH;
    r.a.forEach((v, j) => {
      if (v <= 0) return;
      const xa = X(e[j]) + 0.5, xe = xa + Math.max(0.8, X(e[j + 1]) - X(e[j]) - 1), h = Math.max(0.5, Y(v) * rowH);
      const parts = xs == null || xa >= xs ? [[xa, xe, 1]] : xe <= xs ? [[xa, xe, 0.35]] : [[xa, xs, 0.35], [xs, xe, 1]];
      for (const [p0, p1, op] of parts) if (p1 - p0 > 0.2)
        g += `<rect x="${p0.toFixed(1)}" y="${(base - h).toFixed(1)}" width="${(p1 - p0).toFixed(1)}" height="${h.toFixed(1)}" fill="${r.color}" fill-opacity="${op}"/>`;
    });
    if (r.b) {
      let d = "";
      r.b.forEach((v, j) => {
        const y = (base - Y(v) * rowH).toFixed(1);
        d += `${j ? "L" : "M"}${X(e[j]).toFixed(1)},${y}L${X(e[j + 1]).toFixed(1)},${y}`;
      });
      g += `<path class="step" d="${d}"/>`;
    }
    g += `<line class="base" x1="${L}" x2="${W - R}" y1="${base}" y2="${base}"/>`;
    if (r.marker != null) {
      const x = X(r.marker).toFixed(1);
      g += `<line class="marker" x1="${x}" x2="${x}" y1="${top}" y2="${base}"/>`;
    }
    const mid = top + rowH / 2;
    g += `<text class="rowlab" x="${L - 10}" y="${mid - 2}" text-anchor="end"><tspan class="strong">${r.name[0]}</tspan> ${r.name[1]}</text>`
      + `<text x="${L - 10}" y="${mid + 13}" text-anchor="end">${r.sub}</text>`
      + `<text class="stat" x="${W - R + 10}" y="${mid - 2}">${r.stat[0]}</text>`
      + (r.stat[1] ? `<text x="${W - R + 10}" y="${mid + 13}">${r.stat[1]}</text>` : "");
  });
  if (cfg.vline) {
    const x = X(cfg.vline.x).toFixed(1);
    g += `<line class="vline" x1="${x}" x2="${x}" y1="${T - 8}" y2="${plotBottom}"/>`
      + `<text class="vlab" x="${+x + 5}" y="${T - 10}">${cfg.vline.label}</text>`;
  }
  if (cfg.strip) {
    const y = plotBottom + 22;
    g += `<rect x="${L}" y="${y}" width="${W - R - L}" height="9" fill="url(#grad-${host.id})" stroke="var(--axis)" stroke-width="0.5"/>`
      + `<text x="${L - 10}" y="${y + 8}" text-anchor="end">image grey</text>`;
  }
  g += `<rect class="hband" x="0" y="${T}" width="0" height="${plotBottom - T}" visibility="hidden"/>`
    + `<rect class="hit" x="${L}" y="${T}" width="${W - R - L}" height="${plotBottom - T}"/>`;
  host.innerHTML = `<svg class="chart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${cfg.aria}">${g}</svg>`;
  if (cfg.onScale) cfg.onScale(maxY);

  const svg = host.firstChild, band = svg.querySelector(".hband"), hit = svg.querySelector(".hit");
  hit.addEventListener("mousemove", ev => {
    const v = x0 + (ev.clientX - svg.getBoundingClientRect().left - L) / (W - L - R) * (x1 - x0);
    const j = Math.min(e.length - 2, Math.max(0, Math.floor((v - x0) / bw)));
    band.setAttribute("x", X(e[j]));
    band.setAttribute("width", Math.max(1, X(e[j + 1]) - X(e[j])));
    band.setAttribute("visibility", "visible");
    const span = j === 0 ? `below ${fmt(e[1])}` : j === e.length - 2 ? `${fmt(e[j])} and above` : `${fmt(e[j])} to ${fmt(e[j + 1])}`;
    let html = `<b>${span} a.u.</b><table>`;
    for (const r of rows)
      html += `<tr><td><i class="sw" style="background:${r.color}"></i>${r.name[0]} ${r.name[1]}</td>`
        + `<td>${fmt(r.a[j], cfg.dec)}%${r.b ? ` <span class="muted">· off-tissue ${fmt(r.b[j], cfg.dec)}%</span>` : ""}</td></tr>`;
    tip.show(html + "</table>", ev);
  });
  hit.addEventListener("mouseleave", () => { band.setAttribute("visibility", "hidden"); tip.hide(); });
}

// Pan and zoom of content of size content() = [w, h] inside the element el: wheel, drag, double click to fit and the
// #zin, #zout and #zfit buttons in it. k, tx, ty give the transform; onChange() redraws after every move.
class PanZoom {
  constructor(el, { content, onChange, min = 0.05, max = 8, pad = 1 }) {
    Object.assign(this, { el, content, onChange, min, max, pad, k: 1, tx: 0, ty: 0, fitted: true });
    el.addEventListener("wheel", e => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      this.zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    let drag = null;
    el.addEventListener("pointerdown", e => {
      if (e.target.closest(".zoombtns")) return;
      drag = { x: e.clientX, y: e.clientY, tx: this.tx, ty: this.ty };
      el.classList.add("dragging"); el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointermove", e => {
      if (!drag) return;
      this.fitted = false; this.tx = drag.tx + e.clientX - drag.x; this.ty = drag.ty + e.clientY - drag.y;
      onChange();
    });
    const end = () => { drag = null; el.classList.remove("dragging"); };
    el.addEventListener("pointerup", end); el.addEventListener("pointercancel", end);
    el.addEventListener("dblclick", () => this.fit());
    const centre = f => { const r = el.getBoundingClientRect(); this.zoomAt(f, r.width / 2, r.height / 2); };
    this.centre = centre;
    el.querySelector("#zin").onclick = () => centre(1.6);
    el.querySelector("#zout").onclick = () => centre(1 / 1.6);
    el.querySelector("#zfit").onclick = () => this.fit();
    window.addEventListener("resize", () => this.fitted ? this.fit() : onChange());  // keep a zoom the user chose
  }
  fit() {
    const r = this.el.getBoundingClientRect(), [w, h] = this.content();
    this.k = Math.min(r.width / w, r.height / h) * this.pad;
    this.tx = (r.width - w * this.k) / 2; this.ty = (r.height - h * this.k) / 2; this.fitted = true;
    this.onChange();
  }
  show(x, y, size) {   // fit a square of the content into the view
    const r = this.el.getBoundingClientRect();
    this.k = Math.min(r.width, r.height) / size;
    this.tx = r.width / 2 - (x + size / 2) * this.k; this.ty = r.height / 2 - (y + size / 2) * this.k; this.fitted = false;
    this.onChange();
  }
  zoomAt(f, cx, cy) {
    this.fitted = false;
    const k = Math.min(Math.max(this.k * f, this.min), this.max);
    this.tx = cx - (cx - this.tx) * k / this.k; this.ty = cy - (cy - this.ty) * k / this.k; this.k = k;
    this.onChange();
  }
}

// Display choices kept between visits; get() is {} and set() does nothing where the browser blocks storage.
function persist(key) {
  return {
    get() { try { return JSON.parse(localStorage.getItem(key) || "{}"); } catch (e) { return {}; } },
    set(v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch (e) {} },
  };
}
