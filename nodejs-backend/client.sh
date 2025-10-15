#!/bin/bash

# VLLM Manager Node.js Client Startup Script

echo "🔌 Starting VLLM Manager Client (Node.js Backend)..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    echo "💡 On Ubuntu: sudo apt update && sudo apt install nodejs npm"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "src/client.js" ]; then
    echo "❌ Please run this script from the nodejs-backend directory"
    exit 1
fi

# Start the client
node src/client.js