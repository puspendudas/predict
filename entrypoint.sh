#!/bin/bash
set -e

echo "Starting Prediction Service..."

# Run migration (only migrates if internal DB is empty)
echo "Checking for data migration..."
python /app/migrate.py || echo "Migration step completed (may have warnings)"

# Start the main application
echo "Starting FastAPI application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 7000 --workers 2
