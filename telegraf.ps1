# -----------------------------
# Variables
# -----------------------------
$Version = "1.37.1"
$BaseDir = "C:\Program Files\InfluxData\telegraf"
$ZipName = "telegraf-$Version`_windows_amd64.zip"
$DownloadUrl = "https://dl.influxdata.com/telegraf/releases/$ZipName"
$ZipPath = "$env:TEMP\$ZipName"
$ExtractedDir = "$BaseDir\telegraf-$Version"

# -----------------------------
# Create directories
# -----------------------------
Write-Host "Creating directory $BaseDir"
New-Item -ItemType Directory -Force -Path $BaseDir | Out-Null

# -----------------------------
# Download Telegraf
# -----------------------------
Write-Host "Downloading Telegraf $Version"
Invoke-WebRequest -Uri $DownloadUrl -OutFile $ZipPath

# -----------------------------
# Extract archive
# -----------------------------
Write-Host "Extracting Telegraf"
Expand-Archive -Path $ZipPath -DestinationPath $BaseDir -Force

# -----------------------------
# Move binaries to base folder
# -----------------------------
Write-Host "Moving telegraf.exe and telegraf.conf"
Move-Item "$ExtractedDir\telegraf.exe" $BaseDir -Force
Move-Item "$ExtractedDir\telegraf.conf" $BaseDir -Force

# -----------------------------
# Optional: enable Windows services input
# -----------------------------
$ConfigPath = "$BaseDir\telegraf.conf"
Write-Host "Enabling inputs.win_services plugin (if commented)"
(Get-Content $ConfigPath) `
  -replace '#\s*\[\[inputs.win_services\]\]', '[[inputs.win_services]]' |
  Set-Content $ConfigPath

# -----------------------------
# Install Telegraf as a service
# -----------------------------
Write-Host "Installing Telegraf Windows service"
& "$BaseDir\telegraf.exe" --service install `
  --config "$BaseDir\telegraf.conf"

# -----------------------------
# Test configuration
# -----------------------------
Write-Host "Testing Telegraf configuration"
& "$BaseDir\telegraf.exe" `
  --config "$BaseDir\telegraf.conf" --test

# -----------------------------
# Start service
# -----------------------------
Write-Host "Starting Telegraf service"
& "$BaseDir\telegraf.exe" --service start

Write-Host "Telegraf installation and service startup completed successfully"
