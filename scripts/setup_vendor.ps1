# setup_vendor.ps1 — Download vendor/llama_quantize binaries
# Run once after cloning: .\scripts\setup_vendor.ps1

$Tag = "b9305"
$Zip = "llama-$Tag-bin-win-cpu-x64.zip"
$Url = "https://github.com/ggerganov/llama.cpp/releases/download/$Tag/$Zip"
$Dest = "$PSScriptRoot\..\vendor\llama_quantize"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$ZipPath = Join-Path $env:TEMP $Zip

Write-Host "Downloading $Zip..."
Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing

Write-Host "Extracting to $Dest..."
Expand-Archive -Path $ZipPath -DestinationPath $Dest -Force

Remove-Item $ZipPath
Write-Host "Done. llama-quantize.exe is at $Dest\llama-quantize.exe"
