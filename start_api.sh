#!/bin/bash

# Sales Prediction Pipeline - FastAPI Startup Script
# This script starts the FastAPI server with proper configuration

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Sales Prediction Pipeline${NC}"
echo -e "${BLUE}FastAPI Application${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if Python virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found (.venv)${NC}"
    echo "Please run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if required packages are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import fastapi; import uvicorn" 2>/dev/null || {
    echo -e "${YELLOW}Installing missing dependencies...${NC}"
    pip install -q fastapi uvicorn[standard] python-multipart
}

# Check if transaction data exists
if [ ! -f "transaction.csv" ]; then
    echo -e "${RED}Warning: transaction.csv not found${NC}"
    echo "The API will still work but predictions won't be available"
fi

# Check if models directory exists
mkdir -p models/saved predictions

# Get the current directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo -e "${GREEN}✓ All checks passed${NC}"
echo ""
echo -e "${BLUE}Starting FastAPI Server...${NC}"
echo ""
echo -e "${YELLOW}Dashboard URL:${NC} http://localhost:8000"
echo -e "${YELLOW}API Docs:${NC}     http://localhost:8000/docs"
echo -e "${YELLOW}ReDoc:${NC}        http://localhost:8000/redoc"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the server
cd "$PROJECT_DIR"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
