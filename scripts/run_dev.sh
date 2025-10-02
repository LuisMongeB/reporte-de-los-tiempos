#!/bin/bash

# Development server startup script

echo "🚀 Starting Telegram AI Agent Development Server..."
echo ""

# Set environment
export ENV_FILE=.env.dev

# Check if .env.dev exists
if [ ! -f .env.dev ]; then
    echo "❌ Error: .env.dev file not found!"
    echo "Please create .env.dev from .env.example"
    exit 1
fi

# Check for required environment variables
if ! grep -q "TELEGRAM_BOT_TOKEN" .env.dev; then
    echo "⚠️  Warning: TELEGRAM_BOT_TOKEN not found in .env.dev"
fi

# Start server
echo "📡 Starting FastAPI server on http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
echo ""

uvicorn src.main:app --reload --host 0.0.0.0 --port 8000 --log-level info
