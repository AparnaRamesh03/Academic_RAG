$ErrorActionPreference = "Continue"

function Kill-Port8000 {
    Write-Host "Killing port 8000..."
    Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 5
}

# 1. final_arch
Kill-Port8000
Write-Host "[1/6] Starting final_arch backend..."
Set-Location -Path "c:\Users\juand\Desktop\Academic_RAG\brain\final_arch"
$finalBackend = Start-Process -FilePath "python" -ArgumentList "main.py" -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 20

Write-Host "[2/6] Collecting final_arch results..."
Set-Location -Path "c:\Users\juand\Desktop\Academic_RAG\evaluation"
Remove-Item results_final_arch.json -ErrorAction SilentlyContinue
$env:PYTHONUTF8="1"
python generate_results.py --arch final_arch
Stop-Process -Id $finalBackend.Id -Force -ErrorAction SilentlyContinue

# 2. agentic_scholar
Kill-Port8000
Write-Host "[3/6] Starting agentic_scholar backend..."
Set-Location -Path "c:\Users\juand\Desktop\Academic_RAG\brain\agentic_scholar"
$scholarBackend = Start-Process -FilePath "python" -ArgumentList "main.py" -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 20

Write-Host "[4/6] Collecting agentic_scholar results..."
Set-Location -Path "c:\Users\juand\Desktop\Academic_RAG\evaluation"
Remove-Item results_agentic_scholar.json -ErrorAction SilentlyContinue
$env:PYTHONUTF8="1"
python generate_results.py --arch agentic_scholar
Stop-Process -Id $scholarBackend.Id -Force -ErrorAction SilentlyContinue

# 3. benchmark.py
Write-Host "[5/6] Generating final unified benchmark report..."
python benchmark.py --out-prefix benchmark_final_run > final_benchmark_report.txt

Write-Host "[6/6] DONE! Report saved to evaluation\final_benchmark_report.txt"