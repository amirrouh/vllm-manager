#!/bin/bash

# VLLM Manager Node.js Backend Startup Script

echo "🚀 Starting VLLM Manager Node.js Backend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    echo "💡 On Ubuntu: sudo apt update && sudo apt install nodejs npm"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Please run this script from the nodejs-backend directory"
    exit 1
fi

# Install dependencies if not installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the backend
echo "🔧 Starting backend server..."
npm start