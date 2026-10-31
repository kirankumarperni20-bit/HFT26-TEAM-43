@echo off
echo Starting AI Research Assistant...
start "FastAPI Backend" cmd /k "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"
start "Vite Frontend" cmd /k "cd frontend && npm run dev"
echo.
echo Both backend and frontend servers are launching in separate persistent windows!
echo - Backend API:  http://127.0.0.1:8000
echo - Swagger Docs: http://127.0.0.1:8000/docs
echo - Frontend App: http://localhost:5173
echo - Workspace:    http://localhost:5173/workspace
echo.
pause

