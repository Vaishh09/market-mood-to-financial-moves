# PRISCA Backend Server Startup Script
Write-Host "Starting PRISCA Backend Server..." -ForegroundColor Green
Write-Host "Server will be available at http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Navigate to project directory
Set-Location "c:\Users\Owner\OneDrive\Documents\CORNELL\AI STUDIO PROJECT"

# Start uvicorn server
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
