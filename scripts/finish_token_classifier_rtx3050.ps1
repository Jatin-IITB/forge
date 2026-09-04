# Score the trained token classifier on the FROZEN test set, then package it.
#
# Training is already done. This is the unattended follow-up: it produces the
# only number that counts (test-set quality), and a transferable archive.
#
# Why the frozen test set and not the validation split: training reported
# eval_span_f1 = 0.8796 on data/gold/val.jsonl, but every published Forge
# number -- teacher 0.9482, generative student 0.6360, the G1 ratio -- is
# measured on data/gold/test.jsonl. Comparing across splits would be the exact
# substitution this project has avoided all along, so the classifier is scored
# on test before any claim is made about it.
#
# Both arms run: raw text (no instruction prompt) and the legacy system prompt.
# The raw arm is the one that matters for cost, since dropping ~292 prompt
# tokens is where the throughput target comes from -- but if it costs accuracy,
# that has to be visible rather than assumed away.
#
# QUALITY numbers from this script are hardware-independent and transfer to the
# Mac. SPEED numbers do NOT: run_economics.py carries the Mac's purchase price
# and 22 W draw, so a CUDA throughput figure quoted against it would misstate
# G3. Speed is measured on the Mac, or here with this machine's own price and
# wall draw substituted and labelled as such.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\finish_token_classifier_rtx3050.ps1

param(
    [string]$CheckpointDir = "checkpoints/token_classifier_rtx3050",
    [string]$Gold = "data/gold/test.jsonl",
    [string]$Contract = "contracts/pii_redaction_v2.yaml",
    [string]$TransferDir = "C:\transfer",
    [int]$BatchSize = 8,
    [string]$LogPath = "logs/finish_token_classifier.log"
)

$ErrorActionPreference = "Stop"
$Python = "$PWD\.venv\Scripts\python.exe"
$Merged = Join-Path $CheckpointDir "final-merged"

# PowerShell turns native-command stderr into a TERMINATING error under
# "Stop" once 2>&1 is used, and Python writes warnings and progress bars to
# stderr constantly. Gate on the exit code instead -- that is what actually
# reports failure. This cost a training run earlier today.
function Invoke-Step {
    param([string]$Title, [string[]]$Arguments)

    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    $Previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -u @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    $Code = $LASTEXITCODE
    $ErrorActionPreference = $Previous
    if ($Code -ne 0) {
        throw "$Title failed (exit code $Code). See $LogPath"
    }
}

# --- preconditions, all cheap, all before any GPU work ---------------------
if (-not (Test-Path $Python))  { throw "No venv interpreter at $Python" }
if (-not (Test-Path $Merged))  { throw "No merged model at $Merged -- did training finish?" }
if (-not (Test-Path $Gold))    { throw "Missing $Gold" }

New-Item -ItemType Directory -Force (Split-Path $LogPath -Parent) | Out-Null
New-Item -ItemType Directory -Force "reports/bench" | Out-Null
Set-Content -Path $LogPath -Value "finish_token_classifier run $(Get-Date -Format o)"

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
if (-not $env:HF_HOME) { $env:HF_HOME = "C:\dev\forge-cache\huggingface" }

Write-Host "Checkpoint : $Merged"
Write-Host "Gold       : $Gold"
Write-Host "Log        : $LogPath"

# --- 1. raw text, no instruction prompt (the configuration that matters) ---
Invoke-Step "Inference: raw text input" @(
    "scripts/bench_serving.py",
    "--backend", "token-classifier",
    "--model", $Merged,
    "--device", "cuda",
    "--gold", $Gold,
    "--token-input", "raw",
    "--batch-size", "$BatchSize",
    "--length-bucket",
    "--repeat", "2",
    "--repeat-selection", "first",
    "--config-name", "tc-cuda-raw",
    "--out", "reports/bench/tc_cuda_raw.json",
    "--save-predictions", "data/predictions_tc_cuda_raw.jsonl"
)

Invoke-Step "Gate scoring: raw text (model-only AND system+validators)" @(
    "scripts/run_eval.py", $Gold, "data/predictions_tc_cuda_raw.jsonl",
    "--contract", $Contract, "--ci", "--validators"
)

# --- 2. legacy system prompt, for the accuracy comparison ------------------
Invoke-Step "Inference: legacy system prompt" @(
    "scripts/bench_serving.py",
    "--backend", "token-classifier",
    "--model", $Merged,
    "--device", "cuda",
    "--gold", $Gold,
    "--token-input", "system",
    "--batch-size", "$BatchSize",
    "--length-bucket",
    "--repeat", "2",
    "--repeat-selection", "first",
    "--config-name", "tc-cuda-system",
    "--out", "reports/bench/tc_cuda_system.json",
    "--save-predictions", "data/predictions_tc_cuda_system.jsonl"
)

Invoke-Step "Gate scoring: legacy system prompt" @(
    "scripts/run_eval.py", $Gold, "data/predictions_tc_cuda_system.jsonl",
    "--contract", $Contract, "--ci", "--validators"
)

# --- 3. package -------------------------------------------------------------
Write-Host ""
Write-Host "=== Packaging ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force $TransferDir | Out-Null
$Zip = Join-Path $TransferDir "token_classifier_rtx3050.zip"
if (Test-Path $Zip) { Remove-Item $Zip -Force }

tar -a -cf $Zip -C $CheckpointDir final final-merged train_meta.json
if ($LASTEXITCODE -ne 0) { throw "Packaging failed" }

$Hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower()
$SizeMB = [math]::Round((Get-Item $Zip).Length / 1MB, 1)

# --- 4. summary -------------------------------------------------------------
Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "archive : $Zip"
Write-Host "size    : $SizeMB MB"
Write-Host "sha256  : $Hash"
Write-Host ""
Write-Host "Send these four things back:"
Write-Host "  1. the sha256 above"
Write-Host "  2. reports/bench/tc_cuda_raw.json"
Write-Host "  3. reports/bench/tc_cuda_system.json"
Write-Host "  4. $LogPath"
Write-Host ""
Write-Host "The line that decides whether this ships is system high-severity"
Write-Host "recall: it must remain 1.0000. Faster and worse does not ship."
