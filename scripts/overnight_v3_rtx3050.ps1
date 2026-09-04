# Overnight: retrain the token classifier on the filtered v3 corpus, score it
# on the frozen test set, and package the result.
#
# THE EXPERIMENT
# The shipped token classifier scored micro-F1 0.8845 on the frozen test set,
# trained on data/train_v2.jsonl -- 837 records, 1519 spans. G1 needs 0.9292,
# so it is short by 0.0447.
#
# ADR 0013 established that capacity and data are JOINTLY binding: adding
# capacity alone moved along a frontier rather than toward it. This run holds
# every hyperparameter fixed and changes only the data, so whatever moves is
# attributable to the corpus and nothing else.
#
#     train_v2        837 records   1519 spans   0 teacher-labelled
#     train_v3_clean 1666 records   4863 spans   179 teacher-labelled (Track B)
#
# PRE-REGISTERED PREDICTIONS, recorded before the run per ADR 0013 discipline.
# Report against these even where they fail; a conjunction is what makes a run
# informative, and ADR 0013 was rejected by its own experiment, which is
# considered a good outcome here.
#
#   P1  micro-F1 on the frozen test set rises above 0.8845.
#   P2  STREET_ADDRESS F1 rises above 0.8438. v3 carries 518 STREET_ADDRESS
#       spans against v2's handful, and this type was the generative model's
#       worst at 0.0923 -- it is the clearest test of whether carrier variety
#       teaches fuzzy boundaries.
#   P3  PASSPORT F1 FALLS. v3 contains only 4 PASSPORT spans, so this type is
#       effectively unlearned. Predicted in advance so that a drop is read as
#       the known coverage hole rather than as noise.
#
#   If P1 fails, more data is not the remaining lever and the gap is capacity or
#   task formulation. That is the decision this run exists to make.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\overnight_v3_rtx3050.ps1

param(
    [string]$TrainData   = "data/train_v3_clean.jsonl",
    [string]$ValData     = "data/gold/val.jsonl",
    [string]$Gold        = "data/gold/test.jsonl",
    [string]$OutputDir   = "checkpoints/token_classifier_v3",
    [string]$Contract    = "contracts/pii_redaction_v2.yaml",
    [string]$LogPath     = "logs/overnight_v3.log",
    [string]$TransferDir = "C:\transfer",
    [string]$ExpectedSha = "53174c966654776797736675b29cdbcb83fdb8d351fa2fd48f1978ff49f77903",
    [string]$LearningRate = "2e-4"
)

$ErrorActionPreference = "Stop"
$Python = "$PWD\.venv\Scripts\python.exe"

# PowerShell turns native-command stderr into a TERMINATING error under "Stop"
# once 2>&1 is used, and Python writes warnings to stderr constantly. That cost
# one training run at step zero. Gate on the exit code instead.
function Invoke-Step {
    param([string]$Title, [string[]]$Arguments)
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    $Previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -u @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    $Code = $LASTEXITCODE
    $ErrorActionPreference = $Previous
    if ($Code -ne 0) { throw "$Title failed (exit $Code). See $LogPath" }
}

# --- preconditions: everything cheap, before any GPU time -------------------
if (-not (Test-Path $Python))    { throw "No venv interpreter at $Python" }
if (-not (Test-Path $TrainData)) { throw "Missing $TrainData -- copy it from the Mac" }
if (-not (Test-Path $ValData))   { throw "Missing $ValData" }
if (-not (Test-Path $Gold))      { throw "Missing $Gold" }

$Actual = (Get-FileHash $TrainData -Algorithm SHA256).Hash.ToLower()
if ($Actual -ne $ExpectedSha) {
    throw "train_v3_clean SHA256 mismatch.`n  expected $ExpectedSha`n  actual   $Actual`nA truncated copy would train a quietly worse model."
}

New-Item -ItemType Directory -Force (Split-Path $LogPath -Parent) | Out-Null
New-Item -ItemType Directory -Force "reports/bench" | Out-Null
Set-Content -Path $LogPath -Value "overnight_v3 run $(Get-Date -Format o)"

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
if (-not $env:HF_HOME) { $env:HF_HOME = "C:\dev\forge-cache\huggingface" }

Write-Host "train  : $TrainData (sha ok)"
Write-Host "output : $OutputDir"
Write-Host "log    : $LogPath"

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

# BIOES cannot represent nested spans. A non-zero unsupported count means gold
# labels are silently dropped before training starts, which produces a model
# that trains cleanly and scores badly for reasons you would chase elsewhere.
Invoke-Step "BIOES alignment gate" ($CommonArgs + @("--verify-alignment-only"))

Invoke-Step "Training on filtered v3" $CommonArgs

$Merged = Join-Path $OutputDir "final-merged"
if (-not (Test-Path $Merged)) { throw "No merged model at $Merged" }

Invoke-Step "Inference on the FROZEN test set (raw text, CUDA)" @(
    "scripts/bench_serving.py",
    "--backend", "token-classifier",
    "--model", $Merged,
    "--device", "cuda",
    "--gold", $Gold,
    "--token-input", "raw",
    "--batch-size", "8",
    "--length-bucket",
    "--repeat", "2",
    "--repeat-selection", "first",
    "--machine", "asus-vivobook-pro-15-rtx3050ti",
    "--config-name", "tc-v3-cuda-raw",
    "--out", "reports/bench/tc_v3_cuda_raw.json",
    "--save-predictions", "data/predictions_tc_v3_cuda.jsonl"
)

Invoke-Step "Gate scoring vs the 0.8845 baseline" @(
    "scripts/run_eval.py", $Gold, "data/predictions_tc_v3_cuda.jsonl",
    "--contract", $Contract, "--ci", "--validators"
)

# --- package ----------------------------------------------------------------
Write-Host ""
Write-Host "=== Packaging ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force $TransferDir | Out-Null
$Zip = Join-Path $TransferDir "token_classifier_v3.zip"
if (Test-Path $Zip) { Remove-Item $Zip -Force }
tar -a -cf $Zip -C $OutputDir final final-merged train_meta.json
if ($LASTEXITCODE -ne 0) { throw "Packaging failed" }
$Hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower()

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "archive : $Zip"
Write-Host "sha256  : $Hash"
Write-Host ""
Write-Host "Against the pre-registered predictions:"
Write-Host "  P1  micro-F1 > 0.8845          (baseline trained on 837 records)"
Write-Host "  P2  STREET_ADDRESS F1 > 0.8438"
Write-Host "  P3  PASSPORT F1 falls          (only 4 spans in the corpus)"
Write-Host ""
Write-Host "G1 needs micro-F1 >= 0.9292. System high-severity recall must stay"
Write-Host "1.0000 -- a better F1 does not ship if that floor breaks."
