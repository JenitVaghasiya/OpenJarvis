# jarvis-private.ps1 -- incognito Jarvis session.
#
# Runs Jarvis with OPENJARVIS_HOME pointed at a throwaway temp directory,
# so every runtime artifact (config, traces.db, telemetry.db, audit.db,
# sessions.db, caches, keys) lands in the temp dir -- then deletes it on
# exit. Nothing is written to ~/.openjarvis and no chat content survives
# the window.
#
# Usage:
#   .\jarvis-private.ps1                  # interactive private chat
#   .\jarvis-private.ps1 ask "question"   # one-shot private query
#
# Note: the model runs in Ollama, which holds context in memory only;
# it does not persist prompts to disk.

$ErrorActionPreference = "Stop"

$privateHome = Join-Path $env:TEMP ("jarvis-private-" + [guid]::NewGuid().ToString("N").Substring(0, 12))
New-Item -ItemType Directory -Path $privateHome | Out-Null

# Standalone private config -- everything that can write or phone home is off.
$privateConfig = @'
[engine]
default = "ollama"

[intelligence]
default_model = "qwen3.5:4b"

[agent]
default_agent = "simple"

[server]
host = "127.0.0.1"
port = 8000

[analytics]
enabled = false

[updates]
auto_update = false

[traces]
enabled = false

[telemetry]
enabled = false

[sessions]
enabled = false

[security]
profile = "personal"
'@
# BOM-less UTF-8: Python's tomllib rejects a BOM, and PS 5.1 Out-File -Encoding utf8 writes one.
[System.IO.File]::WriteAllText((Join-Path $privateHome "config.toml"), $privateConfig, (New-Object System.Text.UTF8Encoding($false)))

$env:OPENJARVIS_HOME = $privateHome
$env:OPENJARVIS_NO_UPDATE_CHECK = "1"
$env:Path = "C:\Users\Dev\.local\bin;$env:Path"

Write-Host "[private] session home: $privateHome (wiped on exit)" -ForegroundColor DarkGray

try {
    if ($args.Count -eq 0) {
        uv run jarvis chat
    }
    else {
        uv run jarvis @args
    }
}
finally {
    Remove-Item -Recurse -Force $privateHome -ErrorAction SilentlyContinue
    if (Test-Path $privateHome) {
        Write-Warning "[private] could not fully remove $privateHome -- delete it manually."
    }
    else {
        Write-Host "[private] session wiped. No trace left." -ForegroundColor DarkGray
    }
}
