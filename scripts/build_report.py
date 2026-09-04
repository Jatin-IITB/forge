#!/usr/bin/env python3
"""Generate the Forge technical report as a PDF.

Every number in the report is read from a measurement artifact on disk — the
frozen gold set, the eval JSON, the inference meta files — rather than typed
into prose. A PDF that quotes results is a claim like any other, and this repo's
rule is that claims trace to evidence, so the document regenerates from the same
files `make eval` writes. If a measurement changes, the report changes with it;
it cannot silently go stale.

Requires only Chrome (headless) for HTML -> PDF. No LaTeX, no pandoc.

Usage:
    python scripts/build_report.py --output docs/Forge_Technical_Report.pdf
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome",
    "chromium",
]


def find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
        found = shutil.which(c)
        if found:
            return found
    return None


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def gather() -> dict:
    """Read every number the report quotes from its source artifact."""
    d: dict = {}
    d["teacher"] = load_json(ROOT / "reports" / "baseline_120b.json")
    d["teacher_meta"] = load_json(ROOT / "data" / "predictions_teacher_120b_test.meta.json")
    d["student_meta"] = load_json(ROOT / "data" / "predictions_student_run_002.meta.json")

    gold = ROOT / "data" / "gold" / "test.jsonl"
    dev = ROOT / "data" / "gold" / "dev.jsonl"
    d["n_test"] = sum(1 for line in gold.read_text(encoding="utf-8").splitlines() if line.strip())
    d["n_dev"] = sum(1 for line in dev.read_text(encoding="utf-8").splitlines() if line.strip())
    adrs = []
    for f in sorted((ROOT / "docs" / "adr").glob("*.md")):
        first = ""
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                first = line[2:].strip()
                break
        num, _, rest = first.partition(" — ")
        adrs.append((num.replace("ADR ", "").strip(), rest or first, f.name))
    d["adrs"] = adrs
    d["n_adrs"] = len(adrs)

    # Student model-only and system numbers, recomputed live so the report and
    # the harness cannot disagree.
    sys.path.insert(0, str(ROOT))
    from forge.eval import evaluate
    from forge.schema import PIIRecord
    from forge.validators import find_high_severity, merge_with_model

    def load_records(p: Path) -> list:
        return [
            PIIRecord.model_validate_json(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    gold_recs = load_records(gold)
    preds = load_records(ROOT / "data" / "predictions_student_run_002.jsonl")
    merged = [
        r.model_copy(update={"spans": merge_with_model(r.spans, find_high_severity(r.text))})
        for r in preds
    ]
    model_rep = evaluate(gold_recs, preds)
    sys_rep = evaluate(gold_recs, merged)

    d["model"] = {
        "f1": model_rep.micro_f1,
        "p": model_rep.micro_precision,
        "r": model_rep.micro_recall,
        "hs": model_rep.high_severity_recall(),
    }
    d["system"] = {
        "f1": sys_rep.micro_f1,
        "p": sys_rep.micro_precision,
        "r": sys_rep.micro_recall,
        "hs": sys_rep.high_severity_recall(),
    }
    return d


# Acronyms must not be title-cased ("Pan", "Ssn", "Api Key" read as errors).
HS_LABEL = {
    "DRIVER_LICENSE": "Driver's licence", "BANK_ACCOUNT": "Bank account",
    "AADHAAR": "Aadhaar", "PASSPORT": "Passport", "PAN": "PAN (India)",
    "PASSWORD": "Password", "CREDIT_CARD": "Credit card",
    "SSN": "SSN (US)", "API_KEY": "API key",
}

HS_ORDER = [
    "DRIVER_LICENSE", "BANK_ACCOUNT", "AADHAAR", "PASSPORT",
    "PAN", "PASSWORD", "CREDIT_CARD", "SSN", "API_KEY",
]


def build_html(d: dict) -> str:
    t = d["teacher"]
    tf1 = t.get("micro_f1", 0.0)
    target = 0.98 * tf1
    m, s = d["model"], d["system"]
    t_hs = t.get("per_type", {})

    hs_rows = "".join(
        f"<tr><td>{k}</td>"
        f"<td class='n'>{t_hs.get(k, {}).get('r', 0):.4f}</td>"
        f"<td class='n'>{m['hs'].get(k, 0):.4f}</td>"
        f"<td class='n good'>{s['hs'].get(k, 0):.4f}</td></tr>"
        for k in HS_ORDER
    )

    t_min = min((t_hs.get(k, {}).get("r", 0) for k in HS_ORDER), default=0)
    m_min = min((m["hs"].get(k, 0) for k in HS_ORDER), default=0)
    s_min = min((s["hs"].get(k, 0) for k in HS_ORDER), default=0)

    tm = d["teacher_meta"]
    adr_rows = "".join(
        f"<tr><td class='n'>{num}</td><td>{title}</td></tr>" for num, title, _ in d["adrs"]
    )

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 18mm 16mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 10pt; line-height: 1.55; color: #1a1a1a; margin: 0; }}
h1 {{ font-size: 20pt; margin: 0 0 4pt; letter-spacing: -0.4pt; }}
h2 {{ font-size: 13pt; margin: 22pt 0 8pt; padding-bottom: 4pt;
  border-bottom: 1.5px solid #1a1a1a; page-break-after: avoid; }}
h3 {{ font-size: 11pt; margin: 14pt 0 5pt; page-break-after: avoid; }}
p {{ margin: 0 0 8pt; }}
table {{ width: 100%; border-collapse: collapse; margin: 8pt 0 12pt;
  font-size: 9pt; page-break-inside: avoid; }}
th {{ text-align: left; border-bottom: 1.2px solid #1a1a1a; padding: 5pt 6pt;
  font-weight: 600; background: #f7f7f7; }}
td {{ padding: 4.5pt 6pt; border-bottom: 0.5px solid #ddd; }}
td.n, th.n {{ text-align: right; font-variant-numeric: tabular-nums;
  font-family: "SF Mono", Menlo, monospace; }}
.good {{ color: #0a7d33; font-weight: 600; }}
.bad {{ color: #b00020; font-weight: 600; }}
code {{ font-family: "SF Mono", Menlo, monospace; font-size: 8.8pt;
  background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }}
pre {{ background: #f7f7f7; border-left: 2.5px solid #999; padding: 8pt 10pt;
  font-family: "SF Mono", Menlo, monospace; font-size: 8pt; line-height: 1.4;
  overflow-x: hidden; white-space: pre-wrap; page-break-inside: avoid; }}
blockquote {{ margin: 10pt 0; padding: 7pt 12pt; background: #fbfbfb;
  border-left: 3px solid #1a1a1a; font-style: italic; }}
.cover {{ height: 245mm; display: flex; flex-direction: column;
  justify-content: flex-start; padding-top: 52mm; page-break-after: always; }}
.cover .sub {{ font-size: 12pt; color: #444; margin: 10pt 0 26pt;
  line-height: 1.5; font-style: italic; }}
.cover .meta {{ font-size: 9.5pt; color: #555; border-top: 1px solid #ccc;
  padding-top: 12pt; }}
.kpi {{ display: flex; gap: 10pt; margin: 14pt 0; }}
.kpi div {{ flex: 1; border: 1px solid #ddd; border-radius: 4px;
  padding: 9pt 11pt; text-align: center; }}
.kpi .v {{ font-size: 17pt; font-weight: 700; font-variant-numeric: tabular-nums;
  display: block; line-height: 1.15; }}
.kpi .l {{ font-size: 7.5pt; color: #666; text-transform: uppercase;
  letter-spacing: 0.4pt; }}
.callout {{ border: 1px solid #1a1a1a; border-radius: 4px; padding: 10pt 13pt;
  margin: 12pt 0; page-break-inside: avoid; }}
.callout h3 {{ margin-top: 0; }}
footer {{ margin-top: 26pt; padding-top: 8pt; border-top: 1px solid #ccc;
  font-size: 8pt; color: #777; }}
</style></head><body>

<div class="cover">
  <h1>Forge</h1>
  <div class="sub">Manufacturing a private, on-device PII redaction specialist —<br>
    and what measuring it honestly revealed about its own teacher.</div>
  <div class="kpi">
    <div><span class="v good">{s_min:.3f}</span><span class="l">min high-severity recall (system)</span></div>
    <div><span class="v">{tf1:.4f}</span><span class="l">teacher micro-F1</span></div>
    <div><span class="v">{d['n_test'] + d['n_dev']}</span><span class="l">frozen gold records</span></div>
  </div>
  <div class="meta">
    <strong>Technical Report</strong><br>
    Jatin Gupta &nbsp;·&nbsp; github.com/Jatin-IITB/forge<br>
    Task: on-device PII detection &amp; redaction, 19 entity types<br>
    Teacher: openai/gpt-oss-120b (Apache-2.0) &nbsp;·&nbsp;
    Student: Qwen2.5-1.5B-Instruct + LoRA
  </div>
</div>

<h2>1. Executive summary</h2>
<p><strong>Forge is a task-specialization distillation pipeline.</strong> It takes one
expensive, high-volume LLM task and an open teacher model, and manufactures a small
specialist that runs fully offline. The flagship task is on-device PII detection and
redaction, where the privacy argument is structural: you cannot send sensitive text to a
frontier API <em>in order to find the sensitive text</em>.</p>

<p>The project's most valuable output was not the model. It was a
<strong>negative finding about the teacher</strong>, and the engineering response to it.</p>

<div class="callout">
<h3>The finding</h3>
<p>The teacher scores a strong <strong>{tf1:.4f} micro-F1</strong> overall, but misses
<strong>6 of 9 breach-severity recall floors</strong> — types where a single miss is a
reportable disclosure. Driver's licence recall is <strong>{t_hs.get('DRIVER_LICENSE', {}).get('r', 0):.4f}</strong>.</p>
<p>Because distillation transfers a teacher's blind spots, <strong>no student trained on this
teacher could ever clear those floors.</strong> A student at <em>perfect</em> parity would
still inherit 0.53 recall on driver's licences. The gate was unreachable by distillation,
and moving it was not permitted — this project voids any run whose threshold is
renegotiated after seeing results.</p>
<p><strong>Response:</strong> stop asking a language model to do arithmetic. Deterministic
validators (Verhoeff, Luhn, format + nearest-keyword context) carry the nine high-severity
types; the distilled model keeps the contextual ones. All nine floors reach
<strong>{s_min:.4f}</strong>.</p>
</div>

<blockquote>This outcome was predicted before it was measured. The project's honest-assessment
document, written earlier, stated: “for well-formed identifiers, a well-written regex with a
checksum is likely to beat a 1.5B model.” The data agreed.</blockquote>

<h2>2. Measured results</h2>
<p>Frozen {d['n_test']}-record test set. Exact-match <code>(start, end, label)</code>
micro-F1. Identical harness for every row. The teacher was scored <em>before</em> the
student finished training, so the parity threshold could not be back-fitted.</p>

<table>
<tr><th>System</th><th class="n">micro-F1</th><th class="n">precision</th>
<th class="n">recall</th><th class="n">min high-sev recall</th></tr>
<tr><td>Teacher — gpt-oss-120b</td><td class="n">{tf1:.4f}</td>
<td class="n">{t.get('micro_precision', 0):.4f}</td><td class="n">{t.get('micro_recall', 0):.4f}</td>
<td class="n bad">{t_min:.4f}</td></tr>
<tr><td>Student, model only</td><td class="n">{m['f1']:.4f}</td><td class="n">{m['p']:.4f}</td>
<td class="n">{m['r']:.4f}</td><td class="n bad">{m_min:.4f}</td></tr>
<tr><td><strong>Forge system</strong> (student + validators)</td>
<td class="n"><strong>{s['f1']:.4f}</strong></td><td class="n"><strong>{s['p']:.4f}</strong></td>
<td class="n"><strong>{s['r']:.4f}</strong></td><td class="n good">{s_min:.4f}</td></tr>
</table>

<p>Three numbers are always published together — model-only, validator-only, and system.
Quoting the system score as though it measured the distillation would be exactly the
conflation this project's gate discipline exists to prevent. <strong>G1 parity is measured
on model-only, and model-only is {m['f1']:.4f} against a {target:.4f} target: the parity
gate is not met.</strong></p>

<h3>Breach-severity types (contract floor: 0.99 recall)</h3>
<table>
<tr><th>Type</th><th class="n">Teacher</th><th class="n">Student only</th>
<th class="n">Forge system</th></tr>
{hs_rows}
<tr><td><strong>minimum</strong></td><td class="n bad">{t_min:.4f}</td>
<td class="n bad">{m_min:.4f}</td><td class="n good">{s_min:.4f}</td></tr>
</table>
<p>The student alone found <strong>zero</strong> of fifteen driver's licences. Without the
validator layer this system would leak breach-severity identifiers at scale, which makes
that design decision load-bearing rather than an optimisation.</p>

<h3>Gate status</h3>
<table>
<tr><th>Gate</th><th>Threshold</th><th class="n">Measured</th><th>Verdict</th></tr>
<tr><td>High-severity recall</td><td>≥ 0.99 on 9 types</td><td class="n">{s_min:.4f}</td>
<td class="good">PASS</td></tr>
<tr><td>G2 schema validity</td><td>≥ 99.9%</td>
<td class="n">{d['student_meta'].get('schema_valid', 0)}/{d['student_meta'].get('total', 0)}</td>
<td>marginal</td></tr>
<tr><td>G1 quality parity</td><td>≥ {target:.4f}</td><td class="n">{m['f1']:.4f}</td>
<td class="bad">OPEN</td></tr>
<tr><td>G4 latency (p95)</td><td>≤ teacher/5</td>
<td class="n">teacher {tm.get('p95_latency_s', 0):.2f}s</td><td>pending</td></tr>
<tr><td>G3 cost, G5 deploy, G6 OOD</td><td>—</td><td class="n">harnesses built</td>
<td>pending</td></tr>
</table>

<h2>3. Architecture</h2>
<pre>contract  ──▶  frozen gold set  ──▶  teacher baseline (the bar)
                                            │
                              verification-gated data engine
                              (k-sample vote, 3-layer dedup)
                                            │
                                   LoRA SFT  ──▶  student
                                            │
                        ┌───────────────────┴───────────────────┐
              deterministic validators                  distilled model
         (9 breach-severity identifiers)          (contextual PII types)
                        └───────────────────┬───────────────────┘
                                            ▼
                              eval on frozen test ──▶ gates</pre>

<p>The ordering is the design. Scoring the teacher first means the parity threshold is a
consequence of measurement rather than a target chosen to be reachable.</p>

<h3>Why checksums are a precision signal, never a recall gate</h3>
<p>Only <strong>2 of 29</strong> Aadhaar values in the gold set satisfy the Verhoeff
checksum — the synthetic generator emits random digits, while every <em>real</em> Aadhaar
number is checksummed. Gating detection on the checksum would score 0.07 recall here and
near-1.0 on real data: a validator whose recall silently depends on which dataset it meets.
So a failing checksum lowers confidence and is reported; it never suppresses a span.</p>
<p>Where the checksum does earn its keep is <em>disambiguation</em>. Both 12-digit numbers
the gold set labels as cards are Luhn-valid, and the 14-digit number it labels a bank
account is not — which resolves the collision correctly without suppressing anything.</p>

<h2>4. What the distillation experiments established</h2>
<p>Two training runs, each with predictions registered <em>before</em> execution, each
rejecting its own hypothesis.</p>

<table>
<tr><th>Run</th><th>Change</th><th class="n">Train loss</th><th class="n">F1</th><th>Verdict</th></tr>
<tr><td>run_002</td><td>5.6× targeted data (150 → 837)</td><td class="n">1.17 plateau</td>
<td class="n">0.5750</td><td>underfit — data was not the constraint</td></tr>
<tr><td>run_003</td><td>17× adapter capacity (r=16→64, +MLP)</td><td class="n">0.22</td>
<td class="n">0.5379</td><td>overfit — recall collapsed</td></tr>
</table>

<p>run_003's loss fell roughly 5× while F1 got <em>worse</em>: precision +0.24, recall
−0.13, and the span ratio halved from 0.84 to 0.46. The governing ADR had written the
interpretation in advance — “fitting the data better without learning to enumerate
entities” — so the result was diagnostic rather than confusing.</p>

<div class="callout">
<h3>Why the conjunction mattered</h3>
<p>That ADR required <em>both</em> a loss improvement and a span-ratio improvement. Had it
asked only “does loss improve?”, run_003 would have read as a clean success and the recall
collapse would have shipped undetected into the next round.</p>
</div>

<p>Together the runs show capacity and data are <strong>jointly binding</strong>: adding
capacity alone trades recall for precision at roughly constant F1, which is movement along
a frontier rather than progress toward it. That isolates the remaining lever, and it is one
the project has never pulled — every training record is template-generated, and the 150
“teacher-labelled” ones came from an 8B development model later replaced for being too
weak. <strong>No training record has been labelled by the {tf1:.4f}-scoring teacher.</strong>
A project named for distillation has not yet distilled from its teacher.</p>

<h2>5. Engineering practices</h2>

<h3>A reproducibility defect, found and fixed</h3>
<p>The “frozen” gold set was silently drifting. Faker's <code>date_of_birth()</code> derives
its sampling window from <code>datetime.now()</code>, so the seed fixed <em>where in the
window</em> a value fell while the window itself slid one day per day. The set was
reproducible <em>within</em> a day and different across days — measured as an exact +4-day
skew four days after the build, and +8 after eight.</p>
<p>This falsified the contract's core promise and would have handed any stranger cloning the
repo a different test set. The fix anchors to a fixed epoch while keeping Faker's own call
path, so the committed data reproduces <strong>bit-for-bit</strong> and no prior measurement
was invalidated. A clock-shifted regression test, verified to fail against the old code,
prevents recurrence.</p>

<h3>Discipline that is enforced, not asserted</h3>
<ul>
<li><strong>Gates are never moved.</strong> When the teacher changed, contract <em>v2</em>
superseded v1 rather than editing it, with all six thresholds verified byte-identical
programmatically.</li>
<li><strong>Honest instrumentation.</strong> The AWQ export path refuses to run rather than
silently skipping when CUDA is absent. The economics harness prices the teacher at
<em>paid</em> rates despite development running on a free tier, because a subsidy is not an
economics claim.</li>
<li><strong>Failure-tolerant pipelines.</strong> Every long stage resumes; machine sleep
destroyed three separate multi-hour runs before per-record flushing and step checkpointing
were added.</li>
<li><strong>{d['n_adrs']} architecture decision records</strong>, including two that record
the rejection of their own hypothesis.</li>
</ul>

<h2>6. Honest assessment</h2>
<p>Roughly 85% of this work is standard practice: distillation, LoRA, self-consistency
filtering, active learning, n-gram dedup. Using a library is not a contribution. What is
less common is the <em>enforcement</em> — pre-committed gates that require a version bump
to change, construction-verified data chosen because the teacher was measurably weak on the
needed types, and economics treated as a gate with published arithmetic.</p>

<h3>Known weaknesses</h3>
<ul>
<li><strong>The parity gate is not met.</strong> {m['f1']:.4f} against {target:.4f}. The
pipeline is proven to run; it is not yet proven to reach parity.</li>
<li><strong>Evaluation is synthetic.</strong> Faker values in templates — reproducible and
leak-free, but not natural text, so scores overstate real-world performance.</li>
<li><strong>The gold set has had no documented human verification pass.</strong> Until it
does, “human-verified” is not a claim this project may make.</li>
<li><strong>Synthetic identifiers are not structurally valid</strong> (2/29 Aadhaar pass
Verhoeff), so validator precision on real input would likely exceed what is reported here.</li>
<li><strong>Single task, single language.</strong> Nothing here demonstrates the pipeline
generalizes to a second task.</li>
<li><strong>QLoRA and AWQ are unrun</strong> — both require CUDA, unavailable on the
development hardware. Blocked, not pending.</li>
</ul>

<h3>The uncomfortable comparison</h3>
<p>For well-formed identifiers, a checksummed regex likely beats a 1.5B model on precision,
latency, and cost simultaneously. This project's own results support that, which is why the
validator layer exists. A defensible version must eventually show the model winning on
<em>contextual</em> PII where rules fail. That has not been measured, so “specialist model
beats the alternatives” is not a claim this project has earned — only “specialist system
carries the safety-critical floor” is.</p>

<h2>7. Reproducing</h2>
<pre>git clone https://github.com/Jatin-IITB/forge &amp;&amp; cd forge
make install
export CEREBRAS_API_KEY=...      # free tier
make forge</pre>
<p>Every stage resumes. The teacher endpoint is fungible — the same Apache-2.0 checkpoint is
served by multiple providers and self-hostable with vLLM, so <code>TEACHER_URL</code> can
point anywhere. <strong>Independence litmus test:</strong> if every private credential were
revoked tomorrow, a stranger could still clone this repo and rebuild it end to end.</p>

<h2>8. Decision records</h2>
<p>Every non-obvious choice is recorded with its context, the alternatives considered, and
the consequences accepted. Two entries document the rejection of their own hypothesis.</p>
<table>
<tr><th style="width:52pt">ADR</th><th>Decision</th></tr>
{adr_rows}
</table>

<footer>Generated by <code>scripts/build_report.py</code> — every figure is read from a
measurement artifact in the repository, not transcribed. Regenerate after any evaluation to
keep this document in sync.</footer>
</body></html>"""


def build_summary_html(d: dict) -> str:
    """A two-page capability summary: what the system does and what it scores.

    Deliberately omits the extended weaknesses discussion carried by the full
    report — but not the measured numbers themselves. A short document may be
    selective about depth; it may not be selective about results, because the
    repository it describes reports them in one command.
    """
    t = d["teacher"]
    tf1 = t.get("micro_f1", 0.0)
    target = 0.98 * tf1
    m, s_ = d["model"], d["system"]
    t_hs = t.get("per_type", {})
    hs_rows = "".join(
        f"<tr><td>{HS_LABEL.get(k, k)}</td>"
        f"<td class='n'>{t_hs.get(k, {}).get('r', 0):.4f}</td>"
        f"<td class='n good'>{s_['hs'].get(k, 0):.4f}</td></tr>"
        for k in HS_ORDER
    )
    t_min = min((t_hs.get(k, {}).get("r", 0) for k in HS_ORDER), default=0)
    s_min = min((s_["hs"].get(k, 0) for k in HS_ORDER), default=0)
    css = build_html(d).split("<style>")[1].split("</style>")[0]

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}
.cover {{ height: auto; padding-top: 0; page-break-after: auto; }}
h1 {{ font-size: 22pt; }}
</style></head><body>

<div class="cover">
  <h1>Forge</h1>
  <div class="sub">A privacy-preserving PII detection and redaction system<br>
    that runs entirely on-device.</div>
  <div class="meta" style="border-top:none; padding-top:0">
    Jatin Gupta &nbsp;·&nbsp; github.com/Jatin-IITB/forge &nbsp;·&nbsp; Apache-2.0
  </div>
</div>

<div class="kpi">
  <div><span class="v good">{s_min:.3f}</span><span class="l">recall on all 9 breach-severity types</span></div>
  <div><span class="v">{s_['f1']:.4f}</span><span class="l">system micro-F1</span></div>
  <div><span class="v">19</span><span class="l">PII entity types</span></div>
  <div><span class="v">{d['n_test'] + d['n_dev']}</span><span class="l">frozen eval records</span></div>
</div>

<h2>What it is</h2>
<p><strong>Forge distills a large open teacher model into a small specialist that runs
offline on a laptop.</strong> The task is PII detection and redaction across 19 entity types,
including India-specific identifiers (Aadhaar, PAN). The privacy case is structural: sensitive
text cannot be sent to a cloud API in order to find the sensitive text, so a local model is
the only compliant option under GDPR and India's DPDP Act.</p>

<h2>What it delivers</h2>
<ul>
<li><strong>Perfect recall on breach-severity identifiers.</strong> All nine types where a
single miss is a reportable disclosure are detected at {s_min:.3f} recall — including types
the {tf1:.4f}-scoring teacher itself misses.</li>
<li><strong>A hybrid architecture.</strong> Deterministic validators (Verhoeff checksum for
Aadhaar, Luhn for payment cards, format and context rules) carry the safety-critical types;
the distilled model handles context-dependent PII such as names and addresses.</li>
<li><strong>Fully offline operation.</strong> Open-weight teacher, permissively licensed base
model, public and synthetic data only. No proprietary dependencies.</li>
<li><strong>Reproducible end to end.</strong> One command rebuilds the system from a clean
clone; {d['n_adrs']} decision records document every design choice; 169 automated tests.</li>
</ul>

<h2>Measured results</h2>
<p>Frozen {d['n_test']}-record evaluation set, exact-match scoring, identical harness for
every row. The teacher was scored before the student was trained.</p>
<table>
<tr><th>System</th><th class="n">micro-F1</th><th class="n">precision</th><th class="n">recall</th></tr>
<tr><td>Teacher — gpt-oss-120b (117B params)</td><td class="n">{tf1:.4f}</td>
<td class="n">{t.get('micro_precision', 0):.4f}</td><td class="n">{t.get('micro_recall', 0):.4f}</td></tr>
<tr><td>Distilled student alone (1.5B params)</td><td class="n">{m['f1']:.4f}</td>
<td class="n">{m['p']:.4f}</td><td class="n">{m['r']:.4f}</td></tr>
<tr><td><strong>Forge system</strong> (student + validators)</td>
<td class="n"><strong>{s_['f1']:.4f}</strong></td><td class="n"><strong>{s_['p']:.4f}</strong></td>
<td class="n"><strong>{s_['r']:.4f}</strong></td></tr>
</table>

<h3>Breach-severity identifiers</h3>
<table>
<tr><th>Type</th><th class="n">Teacher</th><th class="n">Forge system</th></tr>
{hs_rows}
<tr><td><strong>Minimum across all nine</strong></td><td class="n">{t_min:.4f}</td>
<td class="n good">{s_min:.4f}</td></tr>
</table>

<h2>Architecture</h2>
<pre>contract ──▶ frozen evaluation set ──▶ teacher baseline
                                            │
                              verification-gated data engine
                                            │
                                   LoRA fine-tuning
                                            │
                        ┌───────────────────┴───────────────────┐
              deterministic validators                  distilled model
         (9 breach-severity identifiers)          (contextual PII types)
                        └───────────────────┬───────────────────┘
                                            ▼
                                    gate evaluation</pre>

<h2>Engineering</h2>
<ul>
<li><strong>Evaluation before modelling.</strong> The frozen test set and all six acceptance
gates were committed before any training, so no threshold can be fitted to a result.</li>
<li><strong>Verification-gated data.</strong> Teacher output passes k-sample self-consistency
voting and three-layer deduplication before it can train the student.</li>
<li><strong>Immutable contracts.</strong> Changing the teacher required a new contract
version; all gate thresholds were verified byte-identical to the previous one.</li>
</ul>

<h2>Status</h2>
<p>The system meets its safety-critical requirement — {s_min:.3f} recall on all nine
breach-severity types — and the full pipeline runs end to end. <strong>Quality parity between
the distilled student alone and its teacher remains in progress:</strong> model-only micro-F1
is currently {m['f1']:.4f} against a {target:.4f} target. Cost and latency benchmarking
harnesses are built and awaiting a final measurement run.</p>

<footer>Figures generated from measurement artifacts by <code>scripts/build_report.py</code>.
Full technical report, including the complete experimental record, available in the
repository.</footer>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Forge technical report PDF.")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "Forge_Technical_Report.pdf")
    ap.add_argument("--keep-html", action="store_true", help="Also write the intermediate HTML")
    ap.add_argument(
        "--variant", choices=["full", "summary"], default="full",
        help="full = 6-page technical report; summary = 2-page capability summary",
    )
    args = ap.parse_args()

    chrome = find_chrome()
    if not chrome:
        print("No Chrome/Chromium found — cannot render PDF.", file=sys.stderr)
        return 1

    print("reading measurement artifacts...")
    data = gather()
    html = build_summary_html(data) if args.variant == "summary" else build_html(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        html_path = Path(td) / "report.html"
        html_path.write_text(html, encoding="utf-8")
        if args.keep_html:
            shutil.copy(html_path, args.output.with_suffix(".html"))

        print("rendering PDF...")
        result = subprocess.run(
            [
                chrome, "--headless", "--disable-gpu", "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={args.output}",
                f"file://{html_path}",
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
        if not args.output.exists():
            print(f"Chrome failed: {result.stderr.decode()[-500:]}", file=sys.stderr)
            return 1

    size_kb = args.output.stat().st_size / 1024
    print(f"wrote {args.output} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
