(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v) => (v == null ? "None yet" : `${(v * 100).toFixed(1).replace(/\.0$/, "")}%`);
  const MODES = { dense: "Dense (LSA vectors)", bm25: "Keyword (BM25)", hybrid: "Hybrid (rank fusion)", hybrid_rerank: "Hybrid + rerank (default)" };
  const PROBLEMS = { "refused an answerable question": "Refused, but the docs do answer it", "missing expected fact": "Answered, but missed the key fact",
    "answered an unanswerable question": "Answered something the docs don't cover", "cited wrong document": "Cited the wrong document" };

  const get = (p) => fetch(p).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))));

  get("/evaluation/results").then((e) => {
    const t = e.splits.test, a = t.answers;
    $("intro").textContent = `Measured on ${a.answerable_cases + a.unanswerable_cases} test questions (${a.answerable_cases} the docs answer, ${a.unanswerable_cases} they deliberately don't) that were never used for tuning. Answers come from the ${e.generator} generator; refusal threshold ${e.coverage_threshold}.`;
    const rows = [
      ["Unanswerable questions refused", a.refusal_recall_on_unanswerable, `${Math.round(a.refusal_recall_on_unanswerable * a.unanswerable_cases)} of ${a.unanswerable_cases}`],
      ["Right document in the top 5", t.retrieval.hybrid_rerank.recall_at_5, "retrieval"],
      ["Answers citing an expected document", a.answers_citing_an_expected_document, "grounding"],
      ["Answerable questions answered", a.answer_rate_on_answerable, `${Math.round(a.answer_rate_on_answerable * a.answerable_cases)} of ${a.answerable_cases}`],
      ["Cited documents that were the expected ones", a.citation_precision, "citation precision"],
      ["Expected key facts present in the answer", a.expected_fact_recall, "the weakest step"],
    ];
    $("results").innerHTML = rows.map(([k, v, note]) => `<div class="bar-row"><dt>${k}<span>${esc(note)}</span></dt>
      <dd><span class="track"><span class="fill" style="width:${v * 100}%"></span></span><strong>${pct(v)}</strong></dd></div>`).join("");
    const best = Math.max(...Object.values(t.retrieval).map((r) => r.mrr));
    $("ablation").querySelector("tbody").innerHTML = Object.entries(t.retrieval).map(([k, r]) =>
      `<tr><th scope="row">${MODES[k] || k}</th><td>${pct(r.recall_at_1)}</td><td>${pct(r.recall_at_5)}</td><td class="${r.mrr === best ? "best" : ""}">${r.mrr.toFixed(3)}</td></tr>`).join("");
    $("fails").innerHTML = a.failures.map((f) => `<li><p class="fq">${esc(f.question)}</p><p class="fp">${esc(PROBLEMS[f.problem] || f.problem)}${f.coverage != null ? `, coverage ${Math.round(f.coverage * 100)}%` : ""}</p>
      ${f.answer ? `<p class="fa">${esc(f.answer)}</p>` : ""}</li>`).join("") || "<li>No failures recorded.</li>";
  }).catch(() => { $("intro").textContent = "No evaluation results yet. Run python scripts/evaluate.py, then reload."; });

  get("/metrics").then((m) => {
    const items = [["Questions asked", m.total_queries], ["Refused", pct(m.unanswerable_rate)], ["Median response", `${Math.round(m.p50_latency_ms)} ms`],
      ["95th percentile", `${Math.round(m.p95_latency_ms)} ms`], ["Rated helpful", m.feedback_count ? `${pct(m.helpful_rate)} of ${m.feedback_count}` : "No ratings yet"]];
    $("usage").innerHTML = items.map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join("");
  }).catch(() => { $("usage").innerHTML = "<p>Usage numbers are unavailable. Check that the API is running.</p>"; });
})();
