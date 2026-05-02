#Requires -Version 5.1
<#
.SYNOPSIS
    Flash IDS embedded benchmark to Arduino Mega 2560 or ESP32-C3.

.PARAMETER Device
    mega | esp32c3 | auto | both

.PARAMETER Model
    logreg | tree | mlp | auto

.PARAMETER Port
    COM port override, e.g. -Port COM4

.PARAMETER Monitor
    Open serial monitor after flashing (115200 baud). Press Ctrl+C to exit.

.PARAMETER Collect
    Number of loop cycles to collect, then print paper-ready table.
    Each cycle = 1 ATTACK + 1 NORMAL sample.
    Example: -Collect 20 collects 40 rows total.

.PARAMETER SkipInstall
    Skip arduino-cli / pio install (faster if already set up).

.PARAMETER ShowOutput
    Show full compiler output.

.EXAMPLE
    .\flash.ps1 -Monitor
    .\flash.ps1 -Device mega    -Model logreg  -Monitor
    .\flash.ps1 -Device mega    -Model tree    -Monitor
    .\flash.ps1 -Device esp32c3 -Model logreg  -Monitor
    .\flash.ps1 -Device esp32c3 -Model tree    -Monitor
    .\flash.ps1 -Device esp32c3 -Model mlp     -Monitor
    .\flash.ps1 -Device both    -Monitor
    .\flash.ps1 -Device mega    -Model logreg  -Collect 20
    .\flash.ps1 -Device esp32c3 -Model mlp     -Collect 20
    .\flash.ps1 -SkipInstall    -Monitor
#>
param(
    [ValidateSet("mega","esp32c3","auto","both")]
    [string]$Device = "auto",

    [ValidateSet("logreg","tree","mlp","mlp_f64","xgb","lgb","auto")]
    [string]$Model = "auto",

    [string]$Port = "",
    [switch]$Monitor,
    [int]$Collect = 0,
    [switch]$SkipInstall,
    [switch]$ShowOutput
)

Set-StrictMode -Version 1
$ErrorActionPreference = "Stop"

function Write-Step { param($m) Write-Host "  >> $m" -ForegroundColor Cyan   }
function Write-OK   { param($m) Write-Host "  OK $m" -ForegroundColor Green  }
function Write-Warn { param($m) Write-Host "  !! $m" -ForegroundColor Yellow }
function Write-Fail { param($m) Write-Host " ERR $m" -ForegroundColor Red    }

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Model -> sketch folder mapping
$SketchMap = @{
    "mega_logreg"       = Join-Path $ScriptDir "ids_mega_logreg"
    "mega_tree"         = Join-Path $ScriptDir "ids_mega_tree"
    "esp32c3_logreg"    = Join-Path $ScriptDir "ids_esp32_logreg"
    "esp32c3_tree"      = Join-Path $ScriptDir "ids_esp32_tree"
    "esp32c3_mlp"       = Join-Path $ScriptDir "ids_esp32_mlp"
    "esp32c3_mlp_f64"   = Join-Path $ScriptDir "ids_esp32_mlp_f64"
    "esp32c3_xgb"       = Join-Path $ScriptDir "ids_esp32_xgb"
    "esp32c3_lgb"       = Join-Path $ScriptDir "ids_esp32_lgb"
}

# PlatformIO environment names
$PIOEnvMap = @{
    "mega_logreg"     = "mega_logreg"
    "mega_tree"       = "mega_tree"
    "esp32c3_logreg"  = "esp32_logreg"
    "esp32c3_tree"    = "esp32_tree"
    "esp32c3_mlp"     = "esp32_mlp"
    "esp32c3_mlp_f64" = "esp32_mlp_f64"
    "esp32c3_xgb"     = "esp32_xgb"
    "esp32c3_lgb"     = "esp32_lgb"
}

$CliDir = Join-Path $env:LOCALAPPDATA "arduino-cli"
$CliExe = Join-Path $CliDir "arduino-cli.exe"

$FQBN_MEGA    = "arduino:avr:mega:cpu=atmega2560"
$FQBN_ESP32C3 = "esp32:esp32:esp32c3:CDCOnBoot=cdc,CPUFreq=160,FlashMode=dio,FlashFreq=80,FlashSize=4M,PartitionScheme=default,DebugLevel=none,EraseFlash=none"
$ESP32_URL    = "https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json"

# ---------------------------------------------------------------------------
# arduino-cli wrappers
# ---------------------------------------------------------------------------
function Run-CLI {
    param([string[]]$A)
    if ($ShowOutput) { & $CliExe @A } else { & $CliExe @A 2>&1 | Out-Null }
    return $LASTEXITCODE
}
function Get-CLIOut { param([string[]]$A); return (& $CliExe @A 2>&1) }

# ---------------------------------------------------------------------------
# Step 1 - Install arduino-cli
# ---------------------------------------------------------------------------
function Ensure-CLI {
    if (-not (Test-Path $CliExe)) {
        $onPath = Get-Command arduino-cli -ErrorAction SilentlyContinue
        if ($onPath) { $script:CliExe = $onPath.Source }
    }
    if (Test-Path $CliExe) {
        $v = (& $CliExe version 2>&1) -join ""
        Write-OK "arduino-cli: $v"
        return
    }

    Write-Step "arduino-cli not found, installing..."
    $wg = Get-Command winget -ErrorAction SilentlyContinue
    if ($wg) {
        Write-Step "Trying winget..."
        winget install --id ArduinoSA.CLI --silent --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
        $onPath = Get-Command arduino-cli -ErrorAction SilentlyContinue
        if ($onPath) { $script:CliExe = $onPath.Source; Write-OK "Installed via winget"; return }
    }

    Write-Step "Downloading from GitHub releases..."
    New-Item -ItemType Directory -Path $CliDir -Force | Out-Null
    $rel   = Invoke-RestMethod "https://api.github.com/repos/arduino/arduino-cli/releases/latest" -UseBasicParsing
    $asset = $rel.assets | Where-Object { $_.name -like "*Windows_64bit.zip" } | Select-Object -First 1
    if ($null -eq $asset) { Write-Fail "Cannot find Windows zip: https://arduino.github.io/arduino-cli/"; exit 1 }
    $zip = Join-Path $env:TEMP "arduino-cli.zip"
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip -UseBasicParsing
    Expand-Archive $zip $CliDir -Force
    Remove-Item $zip
    if (-not (Test-Path $CliExe)) { Write-Fail "Extraction failed. Check $CliDir"; exit 1 }
    Write-OK "arduino-cli installed: $CliExe"
}

# ---------------------------------------------------------------------------
# Step 2 - Cores
# ---------------------------------------------------------------------------
function Ensure-Config {
    Write-Step "Checking config..."
    $urls = (Get-CLIOut @("config","get","board_manager.additional_urls")) -join ""
    if ($urls -notlike "*espressif*") {
        & $CliExe config add board_manager.additional_urls $ESP32_URL 2>&1 | Out-Null
    }
    Write-OK "Config ready"
}

function Ensure-Cores {
    Write-Step "Updating board index..."
    & $CliExe core update-index 2>&1 | Out-Null
    $installed = (Get-CLIOut @("core","list")) -join ""
    if ($installed -notlike "*arduino:avr*") {
        Write-Step "Installing arduino:avr..."
        & $CliExe core install arduino:avr 2>&1 | Out-Null
        Write-OK "arduino:avr installed"
    } else { Write-OK "arduino:avr ready" }
    if ($installed -notlike "*esp32:esp32*") {
        Write-Step "Installing esp32:esp32 (~300 MB, one-time)..."
        & $CliExe core install esp32:esp32 2>&1 | Out-Null
        Write-OK "esp32:esp32 installed"
    } else { Write-OK "esp32:esp32 ready" }
}

# ---------------------------------------------------------------------------
# Step 3 - PlatformIO (for ESP32 + TFLite)
# ---------------------------------------------------------------------------
function Ensure-PIO {
    $pio = Get-Command pio -ErrorAction SilentlyContinue
    if ($pio) { Write-OK "PlatformIO ready"; return }

    Write-Step "PlatformIO not found, installing..."
    $pip = Get-Command pip -ErrorAction SilentlyContinue
    if (-not $pip) { $pip = Get-Command pip3 -ErrorAction SilentlyContinue }
    if (-not $pip) {
        Write-Fail "pip not found. Install Python 3 from https://python.org"
        Write-Fail "Then: pip install platformio"
        exit 1
    }
    & pip install platformio 2>&1 | Out-Null
    if (-not (Get-Command pio -ErrorAction SilentlyContinue)) {
        Write-Fail "pio still not found. Add Python Scripts to PATH and restart PowerShell."
        exit 1
    }
    Write-OK "PlatformIO installed"
}

# ---------------------------------------------------------------------------
# Step 4 - Port detection
# ---------------------------------------------------------------------------
function Get-Boards {
    $raw = (& $CliExe board list --json 2>&1) -join ""
    try { return @(($raw | ConvertFrom-Json).detected_ports) } catch { return @() }
}

function Find-Port {
    param([string]$DevType)
    if ($Port -ne "") { return $Port }
    Write-Step "Detecting port for $DevType..."
    $boards = Get-Boards
    foreach ($b in $boards) {
        $name = if ($b.matching_boards -and $b.matching_boards.Count -gt 0) { $b.matching_boards[0].name } else { "" }
        if ($DevType -eq "mega"    -and $name -like "*Mega*")   { return $b.port.address }
        if ($DevType -eq "esp32c3" -and ($name -like "*ESP32*C3*" -or $name -like "*ESP32C3*" -or $name -like "*esp32c3*")) { return $b.port.address }
    }
    return $null
}

function Prompt-Port {
    param([string]$DevType)
    Write-Host ""
    Write-Warn "Cannot auto-detect port for $DevType. Visible ports:"
    $boards = Get-Boards
    if ($boards.Count -gt 0) {
        foreach ($b in $boards) {
            $name = if ($b.matching_boards -and $b.matching_boards.Count -gt 0) { $b.matching_boards[0].name } else { "" }
            Write-Host "    $($b.port.address)  $name" -ForegroundColor Gray
        }
    } else {
        $wmi = Get-WmiObject Win32_PnPEntity -Filter "Name LIKE '%(COM%'" -ErrorAction SilentlyContinue
        if ($wmi) { foreach ($p in $wmi) { Write-Host "    $($p.Name)" -ForegroundColor Gray } }
    }
    Write-Host ""
    Write-Host "  Enter COM port (e.g. COM3):" -ForegroundColor Yellow
    return (Read-Host "  Port").Trim()
}

# ---------------------------------------------------------------------------
# Step 5a - Compile + Upload via arduino-cli (Mega only)
# ---------------------------------------------------------------------------
function Invoke-ArduinoUpload {
    param([string]$Fqbn, [string]$SketchDir, [string]$PortAddr, [string]$Label)

    Write-Step "Compiling $Label..."
    $bd  = Join-Path $env:TEMP "ids_build_$Label"
    New-Item -ItemType Directory -Path $bd -Force | Out-Null
    $out = & $CliExe compile --fqbn $Fqbn --build-path $bd $SketchDir 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Compile failed:"
        Write-Host ($out -join "`n") -ForegroundColor Red
        return $false
    }
    Write-OK "Compiled"

    Write-Step "Uploading to $PortAddr..."
    $out = & $CliExe upload --fqbn $Fqbn --port $PortAddr --input-dir $bd $SketchDir 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Upload failed:"
        Write-Host ($out -join "`n") -ForegroundColor Yellow
        return $false
    }
    Write-OK "Upload complete"
    return $true
}

# ---------------------------------------------------------------------------
# Step 5b - Compile + Upload via PlatformIO (all ESP32 targets)
# ---------------------------------------------------------------------------
function Invoke-PIOUpload {
    param([string]$PIOEnv, [string]$PortAddr, [string]$Label)

    Write-Step "PlatformIO: building $PIOEnv..."
    $pioArgs = @("run", "-e", $PIOEnv, "-t", "upload")
    if ($PortAddr -ne "") { $pioArgs += @("--upload-port", $PortAddr) }

    Push-Location $ScriptDir
    try {
        # Use Start-Process to avoid PowerShell treating pio stderr as NativeCommandError.
        # PlatformIO writes progress (git clone, download) to stderr even on success.
        if ($ShowOutput) {
            & pio @pioArgs
            $rc = $LASTEXITCODE
        } else {
            $tmpOut = [System.IO.Path]::GetTempFileName()
            $tmpErr = [System.IO.Path]::GetTempFileName()
            $proc = Start-Process -FilePath "pio" -ArgumentList $pioArgs `
                -Wait -PassThru -NoNewWindow `
                -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr
            $rc = $proc.ExitCode
            if ($rc -ne 0) {
                Write-Fail "PIO failed (exit $rc):"
                Get-Content $tmpOut | Select-Object -Last 30 | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
                Get-Content $tmpErr | Select-Object -Last 20 | ForEach-Object { Write-Host "  [err] $_" -ForegroundColor Yellow }
                Write-Host ""
                Write-Warn "ESP32-C3 boot mode hint:"
                Write-Host "  1. Hold BOOT" -ForegroundColor Yellow
                Write-Host "  2. Press+release RESET" -ForegroundColor Yellow
                Write-Host "  3. Release BOOT" -ForegroundColor Yellow
                Write-Host "  4. Re-run script" -ForegroundColor Yellow
                Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
                Pop-Location
                return $false
            } else {
                # Show last few lines of output so user sees upload confirmation
                Get-Content $tmpOut | Select-Object -Last 5 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
            }
            Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
        }
        if ($rc -ne 0) { Pop-Location; return $false }
    } finally {
        Pop-Location
    }
    Write-OK "PIO upload complete"
    return $true
}

# ---------------------------------------------------------------------------
# Step 6 - Serial monitor with CSV collection (.NET SerialPort, no extra tools)
# ---------------------------------------------------------------------------
function Open-Monitor {
    param([string]$PortAddr, [int]$NSamples = 0)

    Write-Host ""
    if ($NSamples -gt 0) {
        Write-Host "  Collecting $NSamples cycles on $PortAddr @ 115200 baud..." -ForegroundColor DarkCyan
    } else {
        Write-Host "  Serial Monitor $PortAddr @ 115200 baud" -ForegroundColor DarkCyan
    }
    Write-Host "  Press Ctrl+C to exit" -ForegroundColor Gray
    Write-Host ""

    $sp   = $null
    $rows = [System.Collections.Generic.List[hashtable]]::new()
    $meta = @{}

    try {
        $sp = New-Object System.IO.Ports.SerialPort($PortAddr, 115200, "None", 8, "One")
        $sp.ReadTimeout = 500
        $sp.NewLine     = "`n"
        $sp.Open()
        # Reset the board by toggling DTR: low->high->low
        # This triggers ESP32 reset so we catch full setup() output
        $sp.DtrEnable = $true;  Start-Sleep -Milliseconds 50
        $sp.DtrEnable = $false; Start-Sleep -Milliseconds 50
        $sp.DtrEnable = $true
        # Wait for board to boot (Arduino ~0.5s, ESP32 ~1s)
        Start-Sleep -Milliseconds 1500
        $sp.DiscardInBuffer()   # flush any boot garbage before first line

        $t0 = Get-Date; $gotData = $false

        while ($true) {
            if ($NSamples -gt 0 -and $rows.Count -ge ($NSamples * 2)) { break }
            try {
                $line    = $sp.ReadLine().TrimEnd()
                $gotData = $true

                # Metadata: KEY=VALUE without comma
                if ($line -match '^([A-Z][A-Z0-9_]*)=(.+)$' -and $line -notmatch ',') {
                    $meta[$Matches[1]] = $Matches[2]
                    Write-Host "  [$($Matches[1])] $($Matches[2])" -ForegroundColor DarkGray
                    continue
                }

                if ($line -eq "READY" -or $line -eq "META_REFRESH" -or $line -like "HEADER:*" -or $line -like "WARN:*") {
                    Write-Host "  $line" -ForegroundColor DarkGray; continue
                }

                # CSV: label,pred,prob,latency_us,sram_bytes,correct
                $parts = $line -split ","
                if ($parts.Count -ge 5) {
                    try {
                        $row = @{
                            label      = $parts[0]
                            pred       = [int]$parts[1]
                            prob       = [double]$parts[2]
                            latency_us = [int]$parts[3]
                            sram_bytes = [int]$parts[4]
                            correct    = if ($parts.Count -gt 5) { [int]$parts[5] } else { -1 }
                        }
                        $rows.Add($row)
                        $ok = if ($row.correct -eq 1) { "[OK]" } elseif ($row.correct -eq 0) { "[!!]" } else { "[?]" }
                        Write-Host ("  {0} {1,-6} pred={2} prob={3:F4}  {4} us  sram:{5} B" -f `
                            $ok, $row.label, $row.pred, $row.prob, $row.latency_us, $row.sram_bytes)
                    } catch { Write-Host "  $line" }
                } else {
                    Write-Host "  $line"
                }

            } catch [System.TimeoutException] {
                if ((-not $gotData) -and ((Get-Date) - $t0).TotalSeconds -gt 8) {
                    Write-Warn "No output after 8s -- press Reset on the board"
                    $t0 = Get-Date
                }
            }
        }
    } catch {
        $msg = "$_"
        if ($msg -notlike "*pipeline*" -and $msg -notlike "*stopped*") {
            Write-Warn "Monitor closed: $msg"
        }
    } finally {
        if ($null -ne $sp -and $sp.IsOpen) { $sp.Close() }
        Write-Host ""; Write-Host "  Monitor closed." -ForegroundColor Gray
    }

    if ($rows.Count -gt 0 -and $NSamples -gt 0) {
        Print-Report -Rows $rows -Meta $meta
        Save-Results -Rows $rows -Meta $meta
    }
}

# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
function Save-Results {
    param(
        [System.Collections.Generic.List[hashtable]]$Rows,
        [hashtable]$Meta
    )

    $board   = if ($Meta["BOARD"]) { $Meta["BOARD"] } else { "unknown" }
    $model   = if ($Meta["MODEL"]) { $Meta["MODEL"] } else { "unknown" }
    $ts      = Get-Date -Format "yyyyMMdd_HHmmss"
    $outDir  = Join-Path $ScriptDir "results"
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

    # Raw data CSV
    $csvFile = Join-Path $outDir "${ts}_${board}_${model}.csv"
    $header  = "label,pred,prob,latency_us,sram_bytes,correct"
    $header | Out-File -FilePath $csvFile -Encoding utf8
    foreach ($r in $Rows) {
        "$($r.label),$($r.pred),$($r.prob),$($r.latency_us),$($r.sram_bytes),$($r.correct)" |
            Out-File -FilePath $csvFile -Append -Encoding utf8
    }

    # Summary line appended to master log
    $logFile = Join-Path $outDir "benchmark_log.csv"
    if (-not (Test-Path $logFile)) {
        "timestamp,board,model,f1,roc_auc,size_kb,n_samples,accuracy,lat_mean_us,lat_std_us,lat_min_us,lat_max_us,sram_used_b,sram_limit_b" |
            Out-File -FilePath $logFile -Encoding utf8
    }

    $all_lat  = $Rows | ForEach-Object { $_.latency_us }
    $n        = $Rows.Count
    $acc      = if ($n -gt 0) { ($Rows | Where-Object { $_.correct -eq 1 }).Count / $n } else { 0 }
    $lat_mean = ($all_lat | Measure-Object -Average).Average
    $lat_min  = ($all_lat | Measure-Object -Minimum).Minimum
    $lat_max  = ($all_lat | Measure-Object -Maximum).Maximum
    $lat_std  = 0
    if ($all_lat.Count -gt 1) {
        $sq = $all_lat | ForEach-Object { ($_ - $lat_mean) * ($_ - $lat_mean) }
        $lat_std = [Math]::Sqrt(($sq | Measure-Object -Average).Average)
    }
    $sram_min = ($Rows | ForEach-Object { $_.sram_bytes } | Measure-Object -Minimum).Minimum
    $sram_lim = if ($Meta["SRAM_LIMIT_BYTES"]) { [long]$Meta["SRAM_LIMIT_BYTES"] } else { 0 }
    $sram_used= if ($sram_lim -gt 0) { $sram_lim - $sram_min } else { -1 }
    $f1       = if ($Meta["F1"])       { $Meta["F1"] }       else { "" }
    $roc      = if ($Meta["ROC_AUC"])  { $Meta["ROC_AUC"] }  else { "" }
    $size_kb  = if ($Meta["SIZE_KB"])  { $Meta["SIZE_KB"] }  else { "" }

    $lat_mean_fmt = "{0:F1}" -f $lat_mean
    $lat_std_fmt  = "{0:F1}" -f $lat_std
    $acc_fmt      = "{0:F4}" -f $acc

    "$ts,$board,$model,$f1,$roc,$size_kb,$n,$acc_fmt,$lat_mean_fmt,$lat_std_fmt,$lat_min,$lat_max,$sram_used,$sram_lim" |
        Out-File -FilePath $logFile -Append -Encoding utf8

    # Save human-readable text report
    $txtFile = Join-Path $outDir "${ts}_${board}_${model}_report.txt"
    $report  = @(
        "=" * 60
        "  HARDWARE BENCHMARK RESULTS"
        "=" * 60
        "  Timestamp  : $ts"
        "  Model      : $model"
        "  Board      : $board"
        "  F1 (test)  : $f1     ROC-AUC : $roc"
        "  Size       : $size_kb KB"
        ("  SRAM used  : {0} B ({1:F2} KB)" -f $sram_used, ($sram_used/1024.0))
        ("  Samples    : {0}   Accuracy on hw : {1:F4}" -f $n, $acc)
        ""
        "  Latency per inference:"
        ("    mean = {0} us    std = {1} us" -f $lat_mean_fmt, $lat_std_fmt)
        ("    min  = {0} us    max = {1} us" -f $lat_min, $lat_max)
        ""
        "-" * 60
        "  PAPER-READY BLOCK"
        "-" * 60
        "  Model: $model | Board: $board"
        ("  Inference latency: {0} +/- {1} us (min {2}, max {3})" -f $lat_mean_fmt, $lat_std_fmt, $lat_min, $lat_max)
        ("  SRAM usage: {0} B ({1:F2} KB)" -f $sram_used, ($sram_used/1024.0))
        "  F1 (TON_IoT test set): $f1  |  ROC-AUC: $roc"
        "  Model size: $size_kb KB"
        "-" * 60
    )
    $report | Out-File -FilePath $txtFile -Encoding utf8

    Write-Host ""
    Write-Host "  Saved: $csvFile" -ForegroundColor DarkGreen
    Write-Host "  Log  : $logFile" -ForegroundColor DarkGreen
    Write-Host "  TXT  : $txtFile" -ForegroundColor DarkGreen
}

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
function Print-Report {
    param(
        [System.Collections.Generic.List[hashtable]]$Rows,
        [hashtable]$Meta
    )

    $all_lat  = $Rows | ForEach-Object { $_.latency_us }
    $n_total  = $Rows.Count
    $n_ok     = ($Rows | Where-Object { $_.correct -eq 1 }).Count
    $accuracy = if ($n_total -gt 0) { $n_ok / $n_total } else { 0 }
    $sram_min = ($Rows | ForEach-Object { $_.sram_bytes } | Measure-Object -Minimum).Minimum
    $lat_mean = ($all_lat | Measure-Object -Average).Average
    $lat_min  = ($all_lat | Measure-Object -Minimum).Minimum
    $lat_max  = ($all_lat | Measure-Object -Maximum).Maximum
    $lat_std  = 0
    if ($all_lat.Count -gt 1) {
        $sq = $all_lat | ForEach-Object { ($_ - $lat_mean) * ($_ - $lat_mean) }
        $lat_std = [Math]::Sqrt(($sq | Measure-Object -Average).Average)
    }

    $board   = if ($Meta["BOARD"])        { $Meta["BOARD"] }        else { "?" }
    $model   = if ($Meta["MODEL"])        { $Meta["MODEL"] }        else { "?" }
    $f1      = if ($Meta["F1"])           { $Meta["F1"] }           else { "?" }
    $roc     = if ($Meta["ROC_AUC"])      { $Meta["ROC_AUC"] }      else { "?" }
    $size_kb = if ($Meta["SIZE_KB"])      { $Meta["SIZE_KB"] }      else { "?" }
    $sram_lim= if ($Meta["SRAM_LIMIT_BYTES"]) { [long]$Meta["SRAM_LIMIT_BYTES"] } else { 0 }
    $arena        = if ($Meta["ARENA_USED_BYTES"])  { [int]$Meta["ARENA_USED_BYTES"] }  else { -1 }
    $sram_model_b = if ($Meta["SRAM_MODEL_BYTES"])  { [int]$Meta["SRAM_MODEL_BYTES"] }  else { -1 }
    # Priority: arena (TFLite) > explicit SRAM_MODEL_BYTES > limit-free calculation
    $sram_used = if ($sram_lim -gt 0) { $sram_lim - $sram_min } else { -1 }
    if    ($arena -gt 0)         { $sram_used = $arena }
    elseif ($sram_model_b -ge 0) { $sram_used = $sram_model_b }

    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  HARDWARE BENCHMARK RESULTS" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  Model   : $model"
    Write-Host "  Board   : $board"
    Write-Host "  F1      : $f1     ROC-AUC : $roc"
    Write-Host "  Size    : $size_kb KB"

    if ($arena -gt 0) {
        Write-Host ("  TFLite arena : {0} B ({1:F2} KB)" -f $arena, ($arena/1024.0))
    } elseif ($sram_model_b -ge 0) {
        Write-Host ("  SRAM (model metadata) : {0} B ({1:F2} KB)" -f $sram_model_b, ($sram_model_b/1024.0))
    } elseif ($sram_lim -gt 0) {
        $used = $sram_lim - $sram_min
        if ($used -gt 0) {
            Write-Host ("  SRAM used    : ~{0} B ({1:F2} KB) / {2:F0} KB limit" -f $used, ($used/1024.0), ($sram_lim/1024.0))
        }
    }

    Write-Host ""
    Write-Host ("  Samples : {0}    Accuracy on hw : {1:F4}" -f $n_total, $accuracy)
    Write-Host ""
    Write-Host "  Latency per inference:"
    Write-Host ("    mean = {0:F1} us    std = {1:F1} us" -f $lat_mean, $lat_std)
    Write-Host ("    min  = {0} us       max = {1} us"    -f $lat_min,  $lat_max)
    Write-Host ""

    foreach ($lbl in @("ATTACK","NORMAL")) {
        $sub = $Rows | Where-Object { $_.label -eq $lbl }
        if ($sub.Count -gt 0) {
            $lm = ($sub | ForEach-Object { $_.latency_us } | Measure-Object -Average).Average
            $ac = ($sub | Where-Object { $_.correct -eq 1 }).Count / $sub.Count
            Write-Host ("  [{0}]  lat={1:F1} us    acc={2:F4}   n={3}" -f $lbl, $lm, $ac, $sub.Count)
        }
    }

    Write-Host ""
    Write-Host ("-" * 60) -ForegroundColor DarkCyan
    Write-Host "  PAPER-READY BLOCK" -ForegroundColor DarkCyan
    Write-Host ("-" * 60) -ForegroundColor DarkCyan
    Write-Host "  Model: $model | Board: $board"
    Write-Host ("  Inference latency: {0:F1} +/- {1:F1} us (min {2}, max {3})" -f $lat_mean, $lat_std, $lat_min, $lat_max)
    if ($arena -gt 0) {
        Write-Host ("  TFLite arena: {0} B ({1:F2} KB)" -f $arena, ($arena/1024.0))
    } elseif ($sram_model_b -ge 0) {
        Write-Host ("  SRAM model metadata: {0} B ({1:F2} KB)" -f $sram_model_b, ($sram_model_b/1024.0))
    } elseif ($sram_lim -gt 0 -and $sram_used -gt 0) {
        Write-Host ("  SRAM usage: ~{0} B ({1:F2} KB)" -f $sram_used, ($sram_used/1024.0))
    }
    Write-Host "  F1 (TON_IoT test set): $f1  |  ROC-AUC: $roc"
    Write-Host "  Model size: $size_kb KB"
    Write-Host ("-" * 60) -ForegroundColor DarkCyan
}

# ---------------------------------------------------------------------------
# Flash one device + model
# ---------------------------------------------------------------------------
function Flash-Device {
    param([string]$DevType, [string]$ModelName)

    # Resolve defaults
    if ($ModelName -eq "auto") {
        $ModelName = if ($DevType -eq "mega") { "logreg" } else { "lgb" }
    }
    # mlp/mlp_f64/xgb/lgb only on esp32c3
    if ($DevType -eq "mega" -and ($ModelName -eq "mlp" -or $ModelName -eq "mlp_f64" -or $ModelName -eq "xgb" -or $ModelName -eq "lgb")) {
        Write-Warn "$ModelName requires ESP32-C3. Switching to logreg for Mega."
        $ModelName = "logreg"
    }

    $key        = "${DevType}_${ModelName}"
    $sketchDir  = $SketchMap[$key]
    $pioEnv     = $PIOEnvMap[$key]
    $label      = if ($DevType -eq "mega") { "Arduino Mega 2560" } else { "ESP32-C3 SuperMini" }
    $usesPIO    = ($DevType -eq "esp32c3")

    Write-Host ""
    Write-Host "  ============================================" -ForegroundColor DarkCyan
    Write-Host "  Target  : $label" -ForegroundColor DarkCyan
    Write-Host "  Model   : $ModelName" -ForegroundColor DarkCyan
    $toolset = if ($usesPIO) { "PlatformIO ($pioEnv)" } else { "arduino-cli" }
    Write-Host "  Toolset : $toolset" -ForegroundColor DarkCyan
    Write-Host "  ============================================" -ForegroundColor DarkCyan

    if ($null -eq $sketchDir -or -not (Test-Path $sketchDir)) {
        Write-Fail "Sketch folder not found: $sketchDir"
        return $false
    }

    $p = Find-Port -DevType $DevType
    if ([string]::IsNullOrEmpty($p)) { $p = Prompt-Port -DevType $DevType }
    if ([string]::IsNullOrEmpty($p)) { Write-Warn "No port -- skipping $label"; return $false }
    Write-OK "Port: $p"

    if ($usesPIO) {
        $ok = Invoke-PIOUpload -PIOEnv $pioEnv -PortAddr $p -Label "$label/$ModelName"
    } else {
        $ok = Invoke-ArduinoUpload -Fqbn $FQBN_MEGA -SketchDir $sketchDir -PortAddr $p -Label "$label/$ModelName"
    }
    if (-not $ok) { return $false }

    Start-Sleep -Seconds 2

    if ($Collect -gt 0) {
        Open-Monitor -PortAddr $p -NSamples $Collect
    } elseif ($Monitor) {
        Open-Monitor -PortAddr $p -NSamples 0
    } else {
        Write-OK "Done. Quick monitor:"
        Write-Host "  arduino-cli monitor --port $p --config baudrate=115200" -ForegroundColor Gray
    }
    return $true
}

# ---------------------------------------------------------------------------
# Auto-detect device type
# ---------------------------------------------------------------------------
function Get-AutoDevice {
    $boards = Get-Boards
    foreach ($b in $boards) {
        $name = if ($b.matching_boards -and $b.matching_boards.Count -gt 0) { $b.matching_boards[0].name } else { "" }
        if ($name -like "*Mega*")    { return "mega" }
        if ($name -like "*ESP32*C3*" -or $name -like "*esp32c3*") { return "esp32c3" }
    }
    return "unknown"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "  IDS Embedded -- Hardware Flash" -ForegroundColor Cyan
Write-Host "  Device : $Device   Model : $Model" -ForegroundColor Gray
Write-Host ""

if ($SkipInstall) {
    Write-Warn "SkipInstall -- assuming tools are ready"
    if (-not (Test-Path $CliExe)) {
        $onPath = Get-Command arduino-cli -ErrorAction SilentlyContinue
        if ($onPath) { $script:CliExe = $onPath.Source }
        else { Write-Fail "arduino-cli not found. Remove -SkipInstall."; exit 1 }
    }
} else {
    Ensure-CLI
    Ensure-Config
    Ensure-Cores
    Ensure-PIO
}

$targets = @()
switch ($Device) {
    "mega"    { $targets = @("mega") }
    "esp32c3" { $targets = @("esp32c3") }
    "both"    { $targets = @("mega","esp32c3") }
    "auto"    {
        Write-Step "Auto-detecting connected device..."
        $det = Get-AutoDevice
        if ($det -eq "unknown") {
            $boards = Get-Boards
            if ($boards.Count -gt 0) {
                Write-Host ""
                foreach ($b in $boards) {
                    $name = if ($b.matching_boards -and $b.matching_boards.Count -gt 0) { $b.matching_boards[0].name } else { "" }
                    Write-Host "    $($b.port.address)  $name" -ForegroundColor Gray
                }
                Write-Host ""
                Write-Host "  Which device? (mega / esp32c3):" -ForegroundColor Yellow
                $ans = (Read-Host "  Device").Trim().ToLower()
                if ($ans -eq "mega" -or $ans -eq "esp32c3") { $det = $ans }
                else { Write-Fail "Use -Device mega or -Device esp32c3"; exit 1 }
            } else {
                Write-Warn "No device detected. Plug in and retry."
                exit 1
            }
        }
        Write-OK "Detected: $det"
        $targets = @($det)
    }
}

$allOk = $true
for ($i = 0; $i -lt $targets.Count; $i++) {
    if ($i -gt 0) {
        Write-Host ""
        Write-Host "  Connect next device ($($targets[$i])) and press Enter..." -ForegroundColor Yellow
        Read-Host | Out-Null
    }
    $ok = Flash-Device -DevType $targets[$i] -ModelName $Model
    if (-not $ok) { $allOk = $false }
}

Write-Host ""
if ($allOk) { Write-Host "  All done." -ForegroundColor Green }
else        { Write-Host "  Some steps failed -- see messages above." -ForegroundColor Yellow; exit 1 }
