param([int]$Port = 8788)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python app.py serve --port $Port
