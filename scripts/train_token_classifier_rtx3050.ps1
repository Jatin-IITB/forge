param(
    [string]$TrainData = "data/train_v2.jsonl",
    [string]$ValData = "data/gold/val.jsonl",
    [string]$OutputDir = "checkpoints/token_classifier_rtx3050",
    [string]$LogPath = "logs/token_classifier_rtx3050.log",
    [string]$LearningRate = "2e-4",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$ExpectedTrainSha256 = "d2cf72d639ec69ee4363895290c31ea1a49970200f392d4681b91638ae16eaa0"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Create it and install Forge's [train,dev,data] extras first."
}
if (-not (Test-Path $TrainData)) {
    throw "Missing $TrainData. train_v2 is intentionally gitignored; copy it to this clone."
}
if (-not (Test-Path $ValData)) {
    throw "Missing committed validation split: $ValData"
}

$ActualTrainSha256 = (Get-FileHash $TrainData -Algorithm SHA256).Hash.ToLower()
if ($ActualTrainSha256 -ne $ExpectedTrainSha256) {
    throw "Unexpected train_v2 SHA256: $ActualTrainSha256"
}

& $Python -c @"
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
props = torch.cuda.get_device_properties(0)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {props.total_memory / 2**30:.1f} GiB")
print(f"BF16: {torch.cuda.is_bf16_supported()}")
"@
if ($LASTEXITCODE -ne 0) {
    throw "CUDA preflight failed"
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

Write-Host "Verifying exact BIOES round trips before loading the model..."
& $Python @CommonArgs "--verify-alignment-only"
if ($LASTEXITCODE -ne 0) {
    throw "BIOES alignment verification failed"
}

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$TrainArgs = $CommonArgs
if ($Resume) {
    $TrainArgs += "--resume"
}

New-Item -ItemType Directory -Force (Split-Path $LogPath -Parent) | Out-Null
Write-Host "Starting RTX 3050 QLoRA training..."
& $Python -u @TrainArgs 2>&1 | Tee-Object -FilePath $LogPath
if ($LASTEXITCODE -ne 0) {
    throw "Training failed"
}

Write-Host "Merged classifier: $OutputDir/final-merged"
Write-Host "Training metadata: $OutputDir/train_meta.json"
Write-Host "Training log: $LogPath"
