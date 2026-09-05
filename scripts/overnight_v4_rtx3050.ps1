# Overnight: train on the v2+v3 union, score it, package it.
#
# WHY THIS RUN EXISTS
# The v3 run refuted its own prediction. Model-only micro-F1 on the frozen test
# set went 0.8845 -> 0.8517, backwards, and the per-type table says why:
#
#     high-severity share    v2 38.8%    v3 9.9%    TEST 33.4%
#     PASSPORT               v2 38 sp    v3 4 sp    test 23 sp   -> F1 0.0000
#     STREET_ADDRESS         v2 71 sp    v3 518 sp  test 32 sp   -> F1 0.9412
#
# v3 tripled the span count while ABSOLUTELY REDUCING five of the nine
# high-severity types. It was never "more of v2"; it was a different task
# distribution, 90% prose against a test set that is a third identifiers. So the
# corpus axis moved in mixture, not in amount, and the v3 header's inference
# rule -- "if P1 fails, more data is not the lever, the gap is capacity or task
# formulation" -- was a false dichotomy that skipped the actual explanation.
#
# The union restores identifier density WITHOUT reweighting toward the test
# distribution, which would be a quiet form of peeking at the frozen set. It
# lands at 16.8% high-severity against the test's 33.4% -- deliberately still
# short, so that a gain is attributable to composition rather than to having
# matched the answer key.
#
# The two parents are disjoint: 0 shared texts, 0 gold leakage from either,
# verified on committed bytes by build_train_v4.py, not assumed.
#
#     train_v2          837 rec (816 unique)  1519 spans  38.8% high-sev
#     train_v3_aligned 1662 rec               4847 spans   9.9% high-sev
#     train_v4_aligned 2478 rec               6366 spans  16.8% high-sev
#
# PRE-REGISTERED PREDICTIONS
#   P4  micro-F1 > 0.8845 -- beats BOTH parents. If composition is the
#       mechanism, the union should beat the better parent, not land between.
#   P5  PASSPORT F1 > 0.0000. 42 spans against v3's 4.
#   P6  STREET_ADDRESS F1 >= 0.9412 HOLDS. 589 spans, more than v3 had. If this
#       regresses while P4 succeeds, the types are competing for capacity and
#       the composition account is wrong.
#   P7  CREDIT_CARD recall > 0.4878.
#
#   FALSIFIER: if P4 fails while P5 holds, identifier density was restored and
#   F1 still did not move -- composition is NOT the binding constraint and the
#   remaining gap is capacity or task formulation. That is the fork this run
#   resolves, and it is the claim the failed v3 prediction was not entitled to
#   make on its own.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\overnight_v4_rtx3050.ps1

param(
    [string]$V2           = "data/train_v2.jsonl",
    [string]$V3Aligned    = "data/train_v3_aligned.jsonl",
    [string]$Union        = "data/train_v4.jsonl",
    [string]$AlignedData  = "data/train_v4_aligned.jsonl",
    [string]$AlignedSha   = "8013a6eaaa44736539fb7f093a4f6edd7ff4c0b24496a09db7eb031d77e916e6",
    [string]$ValData      = "data/gold/val.jsonl",
    [string]$Gold         = "data/gold/test.jsonl",
    [string]$OutputDir    = "checkpoints/token_classifier_v4",
    [string]$Contract     = "contracts/pii_redaction_v2.yaml",
    [string]$LogPath      = "logs/overnight_v4.log",
    [string]$TransferDir  = "C:\transfer",
    [string]$LearningRate = "2e-4",
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$Python  = "$PWD\.venv\Scripts\python.exe"
$Machine = "asus-vivobook-pro-15-rtx3050ti"
$Warnings = New-Object System.Collections.ArrayList

# PowerShell turns native-command stderr into a TERMINATING error under "Stop"
# once 2>&1 is used, and Python writes warnings to stderr constantly. That cost
# one training run at step zero. Gate on the exit code, which is what reports
# failure.
function Invoke-Step {
    param([string]$Title, [string[]]$Arguments)
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -u @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { throw "$Title failed (exit $code). See $LogPath" }
}

# --- 1. preconditions -------------------------------------------------------
if (-not (Test-Path $Python))  { throw "No venv interpreter at $Python" }
if (-not (Test-Path $V2))      { throw "Missing $V2" }
if (-not (Test-Path $ValData)) { throw "Missing $ValData" }
if (-not (Test-Path $Gold))    { throw "Missing $Gold" }

New-Item -ItemType Directory -Force (Split-Path $LogPath -Parent) | Out-Null
New-Item -ItemType Directory -Force "reports/bench" | Out-Null
Set-Content -Path $LogPath -Value "overnight_v4 run $(Get-Date -Format o)"

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
if (-not $env:HF_HOME) { $env:HF_HOME = "C:\dev\forge-cache\huggingface" }

Write-Host "output   : $OutputDir"
Write-Host "log      : $LogPath"
Write-Host "started  : $(Get-Date -Format 'HH:mm:ss')"

# --- 2. rebuild the v3 half if it is not already here -----------------------
# The v3 run wrote this file. Regenerating it is seconds and makes this script
# runnable from a clean checkout rather than only as a follow-on.
if (-not (Test-Path $V3Aligned)) {
    if (-not (Test-Path "data/train_v3_clean.jsonl")) {
        throw "Need $V3Aligned or data/train_v3_clean.jsonl to rebuild it from."
    }
    Invoke-Step "Rebuild train_v3_aligned" @(
        "scripts/normalize_spans.py",
        "--in", "data/train_v3_clean.jsonl", "--out", $V3Aligned, "--max-length", "128"
    )
}

# --- 3. build the union, then make it BIOES-representable -------------------
# build_train_v4 checks gold leakage against the committed bytes of BOTH splits
# and refuses to write if it finds any. v2 has never been through the BIOES
# normalizer, so that runs over the union rather than over v3 alone.
Invoke-Step "Build the v2+v3 union" @(
    "scripts/build_train_v4.py", "--v2", $V2, "--v3", $V3Aligned, "--out", $Union
)
Invoke-Step "Normalize spans for BIOES" @(
    "scripts/normalize_spans.py", "--in", $Union, "--out", $AlignedData, "--max-length", "128"
)
if (-not (Test-Path $AlignedData)) { throw "Normalizer wrote nothing to $AlignedData" }

# Generated here rather than copied, so this hash checks that this machine
# derived the same corpus the laptop did. Not fatal: the normalizer re-verifies
# its own output byte-for-byte before exiting, so the file is safe either way.
$alignedActual = (Get-FileHash $AlignedData -Algorithm SHA256).Hash.ToLower()
if ($alignedActual -ne $AlignedSha) {
    [void]$Warnings.Add("union sha differs from the reference ($alignedActual)")
    Write-Host "  NOTE: union sha $alignedActual" -ForegroundColor Yellow
    Write-Host "        reference  $AlignedSha" -ForegroundColor Yellow
    Write-Host "        Self-verified, training anyway. Report this line back." -ForegroundColor Yellow
} else {
    Write-Host "  union matches the reference sha" -ForegroundColor Green
}

# Evaluation integrity: the gold set must need no normalization at all. If this
# ever fires, every number in the ledger is suspect and the run should stop.
Invoke-Step "Gold set needs no normalization" @(
    "scripts/normalize_spans.py", "--in", $Gold, "--check-only", "--max-length", "128"
)

$CommonArgs = @(
    "scripts/train_token_classifier.py",
    "--train-data", $AlignedData,
    "--val-data", $ValData,
    "--base-model", "Qwen/Qwen2.5-1.5B-Instruct",
    "--output-dir", $OutputDir,
    "--full-attention",
    "--qlora",
    "--epochs", "3",
    "--batch-size", "1",
    "--grad-accum", "16",
    "--max-length", "128",
    "--lr", $LearningRate
)

# --- 4. alignment gate, before any GPU time --------------------------------
Invoke-Step "BIOES alignment gate" ($CommonArgs + @("--verify-alignment-only"))

# --- 5. the experiment ------------------------------------------------------
# 2478 records against v3's 1662, same 3 epochs, so expect roughly 2 hours
# rather than 78 minutes.
$Merged = Join-Path $OutputDir "final-merged"
if ($SkipTraining) {
    if (-not (Test-Path $Merged)) {
        throw "-SkipTraining given but no model at $Merged. Drop the switch and train."
    }
    Write-Host ""
    Write-Host "=== Training SKIPPED, scoring the existing model ===" -ForegroundColor Yellow
    Write-Host "  last written: $((Get-Item $Merged).LastWriteTime)" -ForegroundColor Yellow
    [void]$Warnings.Add("training skipped (-SkipTraining); scored a pre-existing model")
} else {
    Invoke-Step "Training on the v2+v3 union" $CommonArgs
    if (-not (Test-Path $Merged)) { throw "No merged model at $Merged" }
}

# --- 6. score on the FROZEN test set ---------------------------------------
Invoke-Step "Inference: frozen test set (raw text, CUDA)" @(
    "scripts/bench_serving.py",
    "--backend", "token-classifier", "--model", $Merged, "--device", "cuda",
    "--gold", $Gold, "--token-input", "raw",
    "--batch-size", "8", "--length-bucket", "--repeat", "2",
    "--repeat-selection", "first", "--machine", $Machine,
    "--hardware-usd", "1024",
    "--config-name", "tc-v4-cuda-raw",
    "--out", "reports/bench/tc_v4_cuda_raw.json",
    "--save-predictions", "data/predictions_tc_v4_cuda.jsonl"
)

Invoke-Step "Gate scoring" @(
    "scripts/run_eval.py", $Gold, "data/predictions_tc_v4_cuda.jsonl",
    "--contract", $Contract, "--ci", "--validators"
)

# --- 7. package -------------------------------------------------------------
Write-Host ""
Write-Host "=== Packaging ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force $TransferDir | Out-Null
$Zip = Join-Path $TransferDir "token_classifier_v4.zip"
if (Test-Path $Zip) { Remove-Item $Zip -Force }
tar -a -cf $Zip -C $OutputDir final final-merged train_meta.json
if ($LASTEXITCODE -ne 0) { throw "Packaging failed" }
$Hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower()

Write-Host ""
Write-Host "=== DONE  $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Green
Write-Host "archive : $Zip"
Write-Host "sha256  : $Hash"
if ($Warnings.Count) {
    Write-Host ""
    Write-Host "Skipped steps:" -ForegroundColor Yellow
    foreach ($w in $Warnings) { Write-Host "  - $w" -ForegroundColor Yellow }
}
Write-Host ""
Write-Host "Against the pre-registered predictions:"
Write-Host "  P4  micro-F1 > 0.8845   (v2 parent; v3 scored 0.8517)"
Write-Host "  P5  PASSPORT F1 > 0.0000"
Write-Host "  P6  STREET_ADDRESS F1 >= 0.9412 holds"
Write-Host "  P7  CREDIT_CARD recall > 0.4878"
Write-Host ""
Write-Host "FALSIFIER: P4 fails while P5 holds -> composition was NOT the"
Write-Host "constraint, and the remaining gap is capacity or task formulation."
Write-Host ""
Write-Host "G1 needs model-only micro-F1 >= 0.9292 (0.98 x teacher 0.9482)."
Write-Host "System high-severity recall must stay 1.0000 -- a better F1 does"
Write-Host "not ship if that floor breaks. Same rule that rejected Q4_K_M."
Write-Host ""
Write-Host "Send back: the micro-F1 block, the MODEL-ONLY vs SYSTEM table, the"
Write-Host "full per-type table, and reports/bench/tc_v4_cuda_raw.json"
