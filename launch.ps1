param([string]$Workspace = (Join-Path $env:LOCALAPPDATA 'stable-audio-local'))
& stable-audio-local --workspace $Workspace launch
exit $LASTEXITCODE
