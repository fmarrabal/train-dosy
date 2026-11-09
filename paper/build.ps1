$ErrorActionPreference = 'Stop'
$paperDirectory = $PSScriptRoot
$sourceDirectory = Join-Path $paperDirectory 'manuscript'
$buildDirectory = Join-Path $paperDirectory 'build'
New-Item -ItemType Directory -Path $buildDirectory -Force | Out-Null
Push-Location $sourceDirectory
try {
    for ($pass = 1; $pass -le 3; $pass++) {
        & pdflatex '-interaction=nonstopmode' '-halt-on-error' '-no-shell-escape' "-output-directory=$buildDirectory" 'Manuscript.tex'
        if ($LASTEXITCODE -ne 0) { throw "LaTeX failed on pass $pass" }
    }
    Copy-Item -LiteralPath (Join-Path $buildDirectory 'Manuscript.pdf') -Destination (Join-Path $paperDirectory 'Manuscript.pdf') -Force
}
finally { Pop-Location }
