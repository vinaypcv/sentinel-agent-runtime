$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$runsRoot = Join-Path $repoRoot ".pytest-runs"
New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss-ffff"
$base = Join-Path $runsRoot "pytest-$PID-$timestamp"
$pytest = Join-Path $repoRoot ".venv\Scripts\pytest.exe"

& $pytest --basetemp="$base" -p no:cacheprovider @args
exit $LASTEXITCODE
