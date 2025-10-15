# VLLM Manager Node.js Backend

A high-performance Node.js backend for the VLLM Manager with WebSocket support for real-time updates.

## Features

- **Real-time Updates**: WebSocket-based live model status updates
- **Better Performance**: Non-blocking I/O for improved responsiveness
- **REST API**: Full REST API for model management
- **Lightweight Client**: Terminal client with minimal resource usage
- **Automatic Monitoring**: Background monitoring of model health and status

## Requirements

- Node.js 14+
- npm
- Python 3.8+ with vLLM installed

## Installation

1. Install Node.js dependencies:
```bash
cd nodejs-backend
npm install
```

## Usage

### Option 1: Using the main VLLM Manager

```bash
# From the main vllm-manager directory
python3 vllm_manager.py nodejs
```

### Option 2: Manual startup

1. Start the backend server:
```bash
cd nodejs-backend
./start.sh
# or
npm start
```

2. In another terminal, start the client:
```bash
cd nodejs-backend
./client.sh
# or
node src/client.js
```

## API Endpoints

- `GET /api/models` - Get all models
- `POST /api/models` - Add a new model
- `PUT /api/models/:name` - Update a model
- `DELETE /api/models/:name` - Delete a model
- `POST /api/models/:name/start` - Start a model
- `POST /api/models/:name/stop` - Stop a model
- `GET /api/gpu` - Get GPU information
- `POST /api/cleanup` - Clean GPU memory

## WebSocket Events

- `models` - Real-time model updates (sent every 5 seconds)

## Performance Benefits

The Node.js backend provides several performance improvements over the Python-only version:

1. **Non-blocking I/O**: Better handling of concurrent operations
2. **Real-time Updates**: WebSocket pushes updates instead of polling
3. **Separation of Concerns**: Backend handles heavy operations, client focuses on UI
4. **Lower Memory Usage**: Terminal client uses minimal resources
5. **Better Responsiveness**: No blocking operations in the UI thread

## Architecture

```
┌─────────────────┐    WebSocket     ┌─────────────────┐    HTTP REST     ┌─────────────────┐
│   Terminal      │ ◄──────────────► │  Node.js        │ ◄──────────────► │   Python vLLM   │
│   Client        │                  │  Backend        │                  │   Processes     │
└─────────────────┘                  └─────────────────┘                  └─────────────────┘
```

## Configuration

The Node.js backend uses the same configuration files as the Python version:
- `~/.vllm-manager/models.json` - Model configurations
- `~/.vllm-manager/.env` - Environment variables

## Troubleshooting

### Backend won't start
- Check that Node.js is installed: `node --version`
- Verify dependencies are installed: `npm install`
- Check that port 3001 is not in use

### Client can't connect
- Ensure the backend server is running
- Check that port 3001 is accessible
- Verify WebSocket connection is not blocked by firewall

### Models not starting
- Verify vLLM is installed in Python
- Check GPU availability
- Ensure required ports are not blocked
- Check backend logs for error messages

## Development

To start the backend in development mode with auto-restart:

```bash
npm run dev
```

This uses `nodemon` to automatically restart the server when files change.