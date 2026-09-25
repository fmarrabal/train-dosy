$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path 'output/pdf' | Out-Null
    for ($pass = 0; $pass -lt 3; $pass++) {
        & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=output/pdf main.tex
        if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed.' }
    }
    Copy-Item -LiteralPath 'output/pdf/main.pdf' -Destination 'output/pdf/Positive_Joint_Laplace_Inversion_Mathematics.pdf' -Force
} finally { Pop-Location }
