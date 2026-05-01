$ErrorActionPreference = "Continue"

function Kill-Port8000 {
    Write-Host "Killing any process on port 8000..."
    Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 5
}

$architectures = @(
    "simple_hybrid_rag",
    "crag_rewrite",
    "self_rag_grader",
    "final_arch",
    "agentic_scholar"
)

$env:PYTHONUTF8="1"

foreach ($arch in $architectures) {
    Write-Host "`n================================================"
    Write-Host "Running Benchmark for Architecture: $arch"
    Write-Host "================================================"
    
    Kill-Port8000
    
    Write-Host "Starting backend for $arch..."
    Set-Location -Path "c:\Users\juand\Desktop\Academic_RAG\brain\$arch"
    $backend = Start-Process -FilePath "python" -ArgumentList "main.py" -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 20
    
    Write-Host "Collecting results for $arch..."
    Set-Location -Path "c:\Users\juand\Desktop\Academic_RAG\evaluation"
    python generate_results.py --arch $arch --benchmark gold_standard_hardcore_40.json
    
    Write-Host "Stopping backend for $arch..."
    Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    Kill-Port8000
}

Write-Host "`n================================================"
Write-Host "Generating Final Hardcore Benchmark Report"
Write-Host "================================================"
Set-Location -Path "c:\Users\juand\Desktop\Academic_RAG\evaluation"
python benchmark.py --results results_simple_hybrid_rag.json results_crag_rewrite.json results_self_rag_grader.json results_final_arch.json results_agentic_scholar.json --out-prefix hardcore_run > hardcore_benchmark_report.txt

Write-Host "DONE! Report saved to hardcore_benchmark_report.txt"
