(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const CATEGORIES = { api: "API", trading: "Trading", account: "Account", security: "Security", operations: "Operations" };
  const SENTENCE = /(?<=[.!?])\s+(?=[A-Z0-9])/;
  const state = { docs: [], byId: {}, current: null, narrowReader: () => matchMedia("(max-width: 760px)").matches };

  async function api(path, opts = {}) {
    const res = await fetch(path, { headers: { "content-type": "application/json" }, ...opts });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    return body;
  }

  // ---------- library ----------
  function family(doc) { return state.docs.filter((d) => d.family === doc.family).sort((a, b) => b.date.localeCompare(a.date)); }

  function renderLibrary() {
    const q = $("filter").value.trim().toLowerCase();
    const shown = state.docs.filter((d) => d.status === "current" || !state.docs.some((o) => o.family === d.family && o.status === "current"))
      .filter((d) => !q || d.title.toLowerCase().includes(q) || d.tags.some((t) => t.toLowerCase().includes(q)));
    const groups = Object.keys(CATEGORIES).map((cat) => [cat, shown.filter((d) => d.category === cat).sort((a, b) => a.title.localeCompare(b.title))]);
    $("shelves").innerHTML = groups.filter(([, docs]) => docs.length).map(([cat, docs]) => `
      <section class="shelf"><h3>${CATEGORIES[cat]}</h3><ul>${docs.map((d) => {
        const n = family(d).length;
        const badge = d.source_type === "incident" ? "Incident" : n > 1 ? `${n} versions` : "";
        return `<li><button type="button" data-doc="${d.document_id}" aria-current="${state.current && family(d).some((v) => v.document_id === state.current)}">${esc(d.title)}${badge ? `<span class="badge">${badge}</span>` : ""}</button></li>`;
      }).join("")}</ul></section>`).join("") || `<p class="quiet">No documents match “${esc(q)}”.</p>`;
  }

  // ---------- reader ----------
  function openReader(docId, highlight = [], sweep = true) {
    const d = state.byId[docId];
    if (!d) return;
    state.current = docId;
    $("reader-empty").hidden = true;
    $("reader-doc").hidden = false;
    $("reader-title").textContent = d.title;
    const newest = family(d)[0];
    const stateText = d.status === "current" ? `<span class="state">Current</span>` : `<span class="state old">Superseded by v${esc(newest.version)}</span>`;
    $("reader-meta").innerHTML = `${CATEGORIES[d.category] || d.category}, version ${esc(d.version)}, ${formatDate(d.date)}. ${stateText}`;
    const versions = family(d);
    $("reader-versions").innerHTML = versions.length > 1
      ? versions.map((v) => `<button type="button" data-version="${v.document_id}" aria-pressed="${v.document_id === docId}">v${esc(v.version)}${v.status === "current" ? " (current)" : ""}</button>`).join("")
        + `<button type="button" data-compare="${versions[0].document_id}|${versions[1].document_id}">Compare v${esc(versions[1].version)} with v${esc(versions[0].version)}</button>`
      : "";
    const marks = new Set(highlight);
    $("reader-body").innerHTML = d.content.split(SENTENCE).map((s, i) =>
      marks.has(i) ? `<mark class="${sweep ? "sweep" : ""}" style="animation-delay:${[...marks].indexOf(i) * 0.25}s">${esc(s)}</mark>` : `<span class="s">${esc(s)}</span>`).join(" ");
    $("reader-tags").innerHTML = d.tags.map((t) => `<span>${esc(t)}</span>`).join("");
    showReader();
    const first = $("reader-body").querySelector("mark"), reader = $("reader");
    reader.scrollTop = first ? Math.max(0, first.offsetTop - reader.clientHeight / 3) : 0;
    renderLibrary();
  }

  async function openCompare(newerId, olderId) {
    try {
      const out = await api(`/documents/${encodeURIComponent(newerId)}/compare/${encodeURIComponent(olderId)}`);
      const then = out.changes.filter((c) => c.op !== "added"), now = out.changes.filter((c) => c.op !== "removed");
      const list = (items) => items.map((c) => `<li class="${c.op === "same" ? "same" : ""}">${numbers(esc(c.text))}</li>`).join("");
      $("reader-title").textContent = `${out.newer.title.replace(/\s*v[\d.]+$/, "")}: what changed`;
      $("reader-meta").innerHTML = `Version ${esc(out.older.version)} (${formatDate(out.older.date)}) against version ${esc(out.newer.version)} (${formatDate(out.newer.date)}). Figures are marked so changes stand out.`;
      $("reader-versions").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(!!b.dataset.compare)));
      $("reader-body").innerHTML = `<div class="compare">
        <section class="then"><h3>Then <span class="quiet">v${esc(out.older.version)}, superseded</span></h3><ol>${list(then)}</ol></section>
        <section class="now"><h3>Now <span class="quiet">v${esc(out.newer.version)}, current</span></h3><ol>${list(now)}</ol></section></div>`;
      $("reader-tags").innerHTML = "";
    } catch (e) { $("reader-body").innerHTML = `<p class="error">Couldn't compare these versions: ${esc(e.message)}</p>`; }
  }

  // Marks quantities (with a unit) and version numbers, the parts that usually change between versions.
  function numbers(html) {
    return html.replace(/\b(\d[\d,.]*(?:-(?:hour|minute|second|day|business-day)|\s?(?:%|minutes?|hours?|days?|business days?|requests?|seconds?))|v\d+\.\d+)(?![\w-])/g, '<span class="num">$1</span>');
  }
  function formatDate(iso) { const d = new Date(iso); return isNaN(d) ? iso : d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }); }

  function showReader() { $("reader").classList.add("open"); if (state.narrowReader()) $("reader").focus(); }
  function hideReader() { $("reader").classList.remove("open"); }

  // ---------- conversation ----------
  function renderAnswer(turn, r) {
    const a = turn.querySelector(".a");
    if (r.insufficient_evidence) {
      const missing = r.missing_terms.length
        ? `The docs say nothing about <span class="missing">${r.missing_terms.map(esc).join(", ")}</span>, so I'm not answering rather than guessing.`
        : "The closest documents don't cover enough of the question to quote an answer.";
      const nearby = r.evidence_coverage >= 0.2 ? r.retrieved.slice(0, 3) : [];
      a.innerHTML = `<div class="refused"><div class="refusal"><p>This isn't covered by the TradeFlow documentation.</p><p class="why">${missing}</p></div>
        ${coverage(r)}
        ${nearby.length ? `<div class="nearby">Closest documents, in case one helps:<ul>${nearby.map((h) => `<li><button type="button" data-doc="${h.document_id}">${esc(h.title)}</button></li>`).join("")}</ul></div>` : ""}</div>`;
    } else {
      a.innerHTML = `<div class="answer-text">${answerHtml(r.answer)}</div>
        <ol class="sources" aria-label="Sources">${r.evidence.map((e, i) => `<li><button type="button" data-cite="${i + 1}">
          <span class="n">${i + 1}</span><span>${esc(e.title)}</span><span class="v">v${esc(e.version)}${e.status === "current" ? "" : ", superseded"}</span></button></li>`).join("")}</ol>
        ${coverage(r)}`;
    }
    a.insertAdjacentHTML("beforeend", `<div class="turn-foot"><span>${generatorLabel(r.generator)}, ${Math.round(r.metrics.total_ms)} ms</span>
      <span role="group" aria-label="Was this helpful?"><button type="button" data-feedback="1">Helpful</button> <button type="button" data-feedback="0">Not helpful</button></span></div>`);
    turn.result = r;
  }

  function coverage(r) {
    const pct = Math.round(r.evidence_coverage * 100), tick = Math.round(r.coverage_threshold * 100);
    return `<div class="coverage"><span>Evidence covers ${pct}% of the question's key terms. Answers need ${tick}%.</span>
      <div class="meter" role="img" aria-label="Coverage ${pct}%, threshold ${tick}%"><span class="fill" style="width:${Math.min(pct, 100)}%"></span><span class="tick" style="left:${tick}%"></span></div></div>`;
  }

  function generatorLabel(g) {
    if (g === "extractive") return "Quoted directly from the documents";
    if (g.startsWith("extractive")) return "Quoted from the documents (the language model was unavailable)";
    if (g === "refusal") return "Refused before generation";
    return `Written by ${g} from the cited documents`;
  }

  function answerHtml(text) {
    const parts = [];
    const re = /(.+?)((?:\s*\[\d+\])+)(?=\s|$)/gs;
    let m, last = 0;
    while ((m = re.exec(text))) {
      const nums = [...m[2].matchAll(/\d+/g)].map((x) => x[0]);
      parts.push(`<span class="s" data-cites="${nums.join(" ")}">${esc(m[1].trim())}</span>${nums.map((n) => `<button type="button" class="cite" data-cite="${n}" aria-label="Open source ${n}">${n}</button>`).join("")}`);
      last = re.lastIndex;
    }
    if (last < text.length) parts.push(esc(text.slice(last)));
    return parts.join(" ");
  }

  function citeClicked(turn, n, button) {
    const ev = turn.result.evidence[n - 1];
    if (!ev) return;
    turn.querySelectorAll(".s.lit").forEach((s) => s.classList.remove("lit"));
    turn.querySelectorAll(`.s[data-cites~="${n}"]`).forEach((s) => s.classList.add("lit"));
    document.querySelectorAll(".cite[aria-pressed=true], .sources button[aria-current=true]").forEach((b) => b.removeAttribute(b.classList.contains("cite") ? "aria-pressed" : "aria-current"));
    if (button.classList.contains("cite")) button.setAttribute("aria-pressed", "true"); else button.setAttribute("aria-current", "true");
    openReader(ev.document_id, ev.highlight);
  }

  async function ask(question) {
    question = question.trim();
    if (!question) return;
    $("welcome")?.remove();
    const turn = $("turn-template").content.firstElementChild.cloneNode(true);
    turn.querySelector(".q").textContent = question;
    turn.querySelector(".a").innerHTML = `<p class="thinking">Searching the documentation…</p>`;
    $("thread").append(turn);
    scrollToTurn(turn);
    $("question").value = ""; autosize();
    const btn = document.querySelector(".ask"); btn.disabled = true;
    try {
      const r = await api("/chat", { method: "POST", body: JSON.stringify({ query: question }) });
      renderAnswer(turn, r);
      scrollToTurn(turn);
      if (!r.insufficient_evidence && r.evidence.length && !state.narrowReader()) {
        citeClicked(turn, 1, turn.querySelector('.sources button[data-cite="1"]'));
      }
    } catch (e) {
      turn.querySelector(".a").innerHTML = `<p class="error">The assistant couldn't answer: ${esc(e.message)}. Check that the API is running, then ask again.</p>`;
    } finally { btn.disabled = false; }
  }

  async function sendFeedback(turn, button) {
    const helpful = button.dataset.feedback === "1";
    try {
      await api("/feedback", { method: "POST", body: JSON.stringify({ request_id: turn.result.request_id, helpful }) });
      button.parentElement.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    } catch (e) { button.textContent = "Couldn't save feedback"; }
  }

  function scrollToTurn(turn) { $("thread").scrollTop = turn.offsetTop - 16; }

  function autosize() { const t = $("question"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 160) + "px"; }

  // ---------- wiring ----------
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); ask($("question").value); });
  $("question").addEventListener("input", autosize);
  $("question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("question").value); } });
  $("examples").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) ask(b.textContent); });
  $("filter").addEventListener("input", renderLibrary);

  $("thread").addEventListener("click", (e) => {
    const turn = e.target.closest(".turn"); if (!turn || !turn.result) return;
    const cite = e.target.closest("[data-cite]"), doc = e.target.closest("[data-doc]"), fb = e.target.closest("[data-feedback]");
    if (cite) citeClicked(turn, +cite.dataset.cite, cite);
    else if (doc) openReader(doc.dataset.doc, []);
    else if (fb) sendFeedback(turn, fb);
  });
  $("shelves").addEventListener("click", (e) => {
    const b = e.target.closest("[data-doc]"); if (!b) return;
    openReader(b.dataset.doc, []); closeLibrary();
  });
  $("reader-versions").addEventListener("click", (e) => {
    const v = e.target.closest("[data-version]"), c = e.target.closest("[data-compare]");
    if (v) openReader(v.dataset.version, []);
    if (c) { const [newer, older] = c.dataset.compare.split("|"); openCompare(newer, older); }
  });
  $("close-reader").addEventListener("click", hideReader);

  function closeLibrary() { $("library").classList.remove("open"); $("open-library").setAttribute("aria-expanded", "false"); }
  $("open-library").addEventListener("click", () => { $("library").classList.add("open"); $("open-library").setAttribute("aria-expanded", "true"); $("filter").focus(); });
  $("close-library").addEventListener("click", closeLibrary);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeLibrary(); hideReader(); } });

  api("/documents").then((docs) => {
    state.docs = docs; state.byId = Object.fromEntries(docs.map((d) => [d.document_id, d]));
    renderLibrary();
  }).catch(() => { $("shelves").innerHTML = `<p class="error">Couldn't load documents. Start the API and reload.</p>`; });
})();
