# Overnight, unattended: retrain on filtered v3, score it, and attempt the
# BigCode external validation. Start it and walk away.
#
# ORDER OF WORK, and why
#   1. preconditions          seconds   fail here rather than at hour two
#   2. BigCode conversion     minutes   OPTIONAL: a gate problem must not kill
#                                       the run that matters
#   3. BIOES alignment gate   seconds   a non-zero count means gold spans are
#                                       silently dropped before training starts
#   4. train on v3            ~1.5-2 h  THE EXPERIMENT
#   5. score on frozen test   minutes   the number that decides G1
#   6. BigCode eval           minutes   OPTIONAL, supplementary
#   7. package + summary
#
# THE EXPERIMENT
# The shipped classifier scored micro-F1 0.8845 on the frozen test set, trained
# on 837 records. G1 needs 0.9292, so it is short by 0.0447. ADR 0013 found
# capacity and data JOINTLY binding, so this run holds every hyperparameter
# fixed and changes only the corpus. Whatever moves is attributable to the data.
#
#     train_v2         837 records  1519 spans    0 teacher-labelled
#     train_v3_clean  1666 records  4863 spans  179 teacher-labelled
#
# PRE-REGISTERED PREDICTIONS (ADR 0013 discipline: report against these even
# where they fail; the conjunction is what makes a run informative).
#   P1  micro-F1 rises above 0.8845.
#   P2  STREET_ADDRESS F1 rises above 0.8438 -- v3 carries 518 spans of it, and
#       this was the generative model's worst type at 0.0923.
#   P3  PASSPORT F1 FALLS. v3 holds only 4 PASSPORT spans, so it is effectively
#       unlearned; predicted now so a drop reads as the known coverage hole
#       rather than as noise.
#   If P1 fails, more data is NOT the remaining lever, and the gap is capacity
#   or task formulation. That is the decision this run exists to make.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\overnight_all_rtx3050.ps1

param(
    [string]$TrainData    = "data/train_v3_clean.jsonl",
    [string]$ValData      = "data/gold/val.jsonl",
    [string]$Gold         = "data/gold/test.jsonl",
    [string]$OutputDir    = "checkpoints/token_classifier_v3",
    [string]$BaselineDir  = "checkpoints/token_classifier_rtx3050",
    [string]$Contract     = "contracts/pii_redaction_v2.yaml",
    [string]$LogPath      = "logs/overnight_all.log",
    [string]$TransferDir  = "C:\transfer",
    [string]$ExpectedSha  = "53174c966654776797736675b29cdbcb83fdb8d351fa2fd48f1978ff49f77903",
    [string]$LearningRate = "2e-4",
    [switch]$SkipBigcode
)

$ErrorActionPreference = "Stop"
$Python  = "$PWD\.venv\Scripts\python.exe"
$Machine = "asus-vivobook-pro-15-rtx3050ti"
$Warnings = New-Object System.Collections.ArrayList

# PowerShell turns native-command stderr into a TERMINATING error under "Stop"
# once 2>&1 is used, and Python writes warnings to stderr constantly. That cost
# one training run at step zero. Gate on the exit code, which is what actually
# reports failure.
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

# Same, but a failure is recorded and the run continues. Used for everything
# supplementary, so a gated dataset or a missing login cannot cost the night.
function Invoke-Optional {
    param([string]$Title, [string[]]$Arguments)
    Write-Host ""
    Write-Host "=== $Title (optional) ===" -ForegroundColor DarkCyan
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -u @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) {
        [void]$Warnings.Add("$Title failed (exit $code) - skipped, run continued")
        Write-Host "  SKIPPED: exit $code. Continuing." -ForegroundColor Yellow
        return $false
    }
    return $true
}

# --- 1. preconditions -------------------------------------------------------
if (-not (Test-Path $Python))    { throw "No venv interpreter at $Python" }
if (-not (Test-Path $TrainData)) { throw "Missing $TrainData -- copy it from the Mac" }
if (-not (Test-Path $ValData))   { throw "Missing $ValData" }
if (-not (Test-Path $Gold))      { throw "Missing $Gold" }

$actual = (Get-FileHash $TrainData -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $ExpectedSha) {
    throw "train_v3_clean SHA256 mismatch.`n  expected $ExpectedSha`n  actual   $actual`nA truncated copy would train a quietly worse model."
}

New-Item -ItemType Directory -Force (Split-Path $LogPath -Parent) | Out-Null
New-Item -ItemType Directory -Force "reports/bench" | Out-Null
New-Item -ItemType Directory -Force "data/external" | Out-Null
Set-Content -Path $LogPath -Value "overnight_all run $(Get-Date -Format o)"

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
if (-not $env:HF_HOME) { $env:HF_HOME = "C:\dev\forge-cache\huggingface" }

Write-Host "train    : $TrainData (sha verified)"
Write-Host "output   : $OutputDir"
Write-Host "log      : $LogPath"
Write-Host "started  : $(Get-Date -Format 'HH:mm:ss')"

# --- 2. BigCode conversion (optional, early so failure is known early) ------
$bigcodeReady = $false
if (-not $SkipBigcode) {
    $bigcodeReady = Invoke-Optional "Convert BigCode PII" @(
        "scripts/load_bigcode_pii.py",
        "--out", "data/external/bigcode_pii.jsonl",
        "--max-chars", "440"
    )
    if (-not $bigcodeReady) {
        Write-Host "  BigCode needs the gate accepted in a browser AND huggingface-cli login." -ForegroundColor Yellow
        Write-Host "  The v3 experiment below is unaffected." -ForegroundColor Yellow
    }
}

$CommonArgs = @(
    "scripts/train_token_classifier.py",
    "--train-data", $TrainData,
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

# --- 3. alignment gate, before any GPU time --------------------------------
# BIOES cannot represent nested spans. A non-zero unsupported count means gold
# labels are dropped before training starts, producing a model that trains
# cleanly and scores badly for reasons you would chase in the wrong place.
Invoke-Step "BIOES alignment gate" ($CommonArgs + @("--verify-alignment-only"))

# --- 4. the experiment ------------------------------------------------------
Invoke-Step "Training on filtered v3" $CommonArgs

$Merged = Join-Path $OutputDir "final-merged"
if (-not (Test-Path $Merged)) { throw "No merged model at $Merged" }

# --- 5. score on the FROZEN test set ---------------------------------------
Invoke-Step "Inference: frozen test set (raw text, CUDA)" @(
    "scripts/bench_serving.py",
    "--backend", "token-classifier", "--model", $Merged, "--device", "cuda",
    "--gold", $Gold, "--token-input", "raw",
    "--batch-size", "8", "--length-bucket", "--repeat", "2",
    "--repeat-selection", "first", "--machine", $Machine,
    "--config-name", "tc-v3-cuda-raw",
    "--out", "reports/bench/tc_v3_cuda_raw.json",
    "--save-predictions", "data/predictions_tc_v3_cuda.jsonl"
)

Invoke-Step "Gate scoring vs the 0.8845 baseline" @(
    "scripts/run_eval.py", $Gold, "data/predictions_tc_v3_cuda.jsonl",
    "--contract", $Contract, "--ci", "--validators"
)

# --- 6. BigCode external validation (optional) ------------------------------
# Runs BOTH models so the external number is comparable rather than isolated.
if ($bigcodeReady -and (Test-Path "data/external/bigcode_pii.jsonl")) {
    foreach ($pair in @(
        @{ Name = "v3";       Model = $Merged },
        @{ Name = "baseline"; Model = (Join-Path $BaselineDir "final-merged") }
    )) {
        if (-not (Test-Path $pair.Model)) { continue }
        $ok = Invoke-Optional "BigCode inference ($($pair.Name))" @(
            "scripts/bench_serving.py",
            "--backend", "token-classifier", "--model", $pair.Model, "--device", "cuda",
            "--gold", "data/external/bigcode_pii.jsonl", "--token-input", "raw",
            "--batch-size", "8", "--length-bucket", "--repeat", "1",
            "--machine", $Machine, "--config-name", "bigcode-$($pair.Name)",
            "--out", "reports/bench/bigcode_$($pair.Name).json",
            "--save-predictions", "data/external/predictions_bigcode_$($pair.Name).jsonl"
        )
        if ($ok) {
            Invoke-Optional "BigCode scoring ($($pair.Name))" @(
                "scripts/run_eval.py", "data/external/bigcode_pii.jsonl",
                "data/external/predictions_bigcode_$($pair.Name).jsonl",
                "--ci", "--validators"
            ) | Out-Null
        }
    }
}

# --- 7. package -------------------------------------------------------------
Write-Host ""
Write-Host "=== Packaging ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force $TransferDir | Out-Null
$Zip = Join-Path $TransferDir "token_classifier_v3.zip"
if (Test-Path $Zip) { Remove-Item $Zip -Force }
tar -a -cf $Zip -C $OutputDir final final-merged train_meta.json
if ($LASTEXITCODE -ne 0) { throw "Packaging failed" }
$Hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower()

# --- 8. summary -------------------------------------------------------------
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
Write-Host "  P1  micro-F1 > 0.8845"
Write-Host "  P2  STREET_ADDRESS F1 > 0.8438"
Write-Host "  P3  PASSPORT F1 falls (only 4 spans in the corpus)"
Write-Host ""
Write-Host "G1 needs micro-F1 >= 0.9292."
Write-Host "System high-severity recall must stay 1.0000 -- a better F1 does not"
Write-Host "ship if that floor breaks. Same rule that rejected Q4_K_M."
Write-Host ""
Write-Host "Send back:"
Write-Host "  1. the micro-F1 block and MODEL-ONLY vs SYSTEM table"
Write-Host "  2. STREET_ADDRESS and PASSPORT rows from the per-type table"
Write-Host "  3. reports/bench/tc_v3_cuda_raw.json"
Write-Host "  4. BigCode output if it ran, including the EXCLUDED percentage"
Write-Host "  5. $LogPath"
