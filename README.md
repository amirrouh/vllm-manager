# 🚀 VLLM Multi-Model Terminal Manager

A clean, organized, and OCD-friendly terminal-based interface for managing multiple VLLM models concurrently with priority-based GPU resource allocation.

## ✨ Features

- **🖥️ Real-time GPU Monitoring**: Live GPU memory, utilization, and temperature tracking
- **🤖 Multi-Model Management**: Run multiple models simultaneously with smart resource allocation
- **⚡ Priority System**: High-priority models automatically free resources by killing lower-priority processes
- **📊 Performance Metrics**: CPU usage, GPU memory, uptime tracking for each model
- **🎨 Cool Terminal UI**: Color-coded status indicators with live updates
- **🔧 CLI Interface**: Command-line tools for automation and scripting

## 🏗️ Architecture

### Clean Code Structure
```
vllm                     # THE ONLY COMMAND YOU NEED
├── src/
│   └── vllm_terminal_manager.py  # Core terminal UI application
├── bin/
│   └── vllm_cli.py             # Command-line interface
├── tests/
│   └── test_terminal_manager.py # Comprehensive test suite
└── vllm-manager.sh            # Legacy simple manager
```

### Data Organization
- **ModelConfig**: Model configuration (HuggingFace ID, port, priority, etc.)
- **ModelState**: Runtime state (status, PID, metrics, health)
- **GPUInfo**: GPU hardware information and utilization
- **ProcessInfo**: Running process details and resource usage

## 🚀 Quick Start

### 1. Launch Terminal Manager
```bash
./vllm
```

### 2. Add Models via CLI
```bash
# Add a high-priority model
./vllm cli add mistral-7b mistralai/Mistral-7B-Instruct-v0.2 --port 8001 --priority 2 --gpu-memory 0.3

# Add a medium-priority model  
./vllm cli add dialog-gpt microsoft/DialoGPT-medium --port 8002 --priority 3 --gpu-memory 0.2

# Add a low-priority background model
./vllm cli add code-llama codellama/CodeLlama-7b-Python-hf --port 8003 --priority 4 --gpu-memory 0.25
```

### 3. Manage Models
```bash
# List all configured models
./vllm cli list

# Start a model
./vllm cli start mistral-7b

# Check system status
./vllm cli status

# Stop a model
./vllm cli stop mistral-7b
```

## 🎮 Terminal UI Controls

| Key | Action |
|-----|---------|
| `↑`/`↓` | Navigate between models |
| `ENTER` | Start/Stop selected model |
| `a` | Add new model |
| `d` | Delete selected model |
| `k` | Kill model process |
| `c` | GPU Cleanup (aggressive) |
| `C` | Nuclear Cleanup (kill all non-critical) |
| `r` | Refresh display |
| `q` | Quit application |

## 🎯 Priority System

| Priority | Description | Behavior |
|----------|-------------|----------|
| 1-2 | High Priority | Protected from automatic termination |
| 3 | Medium Priority | May be killed for higher priority models |
| 4-5 | Low Priority | First to be terminated when memory needed |

## 📊 Status Indicators

- 🟢 **Running**: Model is healthy and accepting requests
- 🟡 **Starting**: Model is initializing 
- 🟠 **Unhealthy**: Model running but health check failing
- 🔴 **Error**: Model failed to start or crashed
- ⚫ **Stopped**: Model is not running

## 🔧 CLI Reference

### Add Model
```bash
python vllm_cli.py add <name> <huggingface_id> --port <port> [options]

Options:
  --priority {1,2,3,4,5}     Priority level (1=highest, 5=lowest)
  --gpu-memory FLOAT         GPU memory utilization (0.1-0.9)
  --max-len INT             Maximum model length
  --tensor-parallel INT     Tensor parallel size
```

### Model Operations
```bash
./vllm cli list              # List all models
./vllm cli start <name>      # Start model
./vllm cli stop <name>       # Stop model
./vllm cli remove <name>     # Remove model configuration
./vllm cli status            # Show system status
```

### GPU Memory Management
```bash
./vllm cli cleanup           # Aggressive GPU cleanup (preserve priority 1)
./vllm cli cleanup --preserve-priority 2  # Preserve priority 1-2 processes
./vllm cli nuclear           # Nuclear cleanup - kill ALL non-critical processes
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_terminal_manager.py
```

Tests cover:
- CLI functionality
- System monitoring
- Model management
- GPU resource allocation
- Configuration persistence

## 📁 Configuration

Models are stored in `models_config.json`:
```json
{
  "models": [
    {
      "name": "mistral-7b",
      "huggingface_id": "mistralai/Mistral-7B-Instruct-v0.2",
      "port": 8001,
      "priority": 2,
      "gpu_memory_utilization": 0.3,
      "max_model_len": 2048,
      "tensor_parallel_size": 1
    }
  ]
}
```

## 🎨 Cool Features

### Real-time Monitoring
- Live GPU memory and utilization graphs
- Process monitoring with CPU/GPU usage
- Temperature monitoring
- Automatic health checks

### Smart Resource Management
- Automatic low-priority process termination
- Memory availability checking before model starts
- Port conflict resolution
- Graceful process shutdown (SIGTERM → SIGKILL)

### User Experience
- Color-coded status indicators
- Responsive keyboard controls
- Status messages with auto-hide
- Clean, organized display layout

## 🛠️ Requirements

- Python 3.8+
- NVIDIA GPU with CUDA
- vLLM installed
- Required packages: `psutil`, `httpx`, `curses`

## 🤝 Usage Examples

### High-Priority Production Model
```bash
python vllm_cli.py add prod-model meta-llama/Llama-2-7b-chat-hf --port 8000 --priority 1 --gpu-memory 0.4
python vllm_cli.py start prod-model
```

### Development/Testing Models
```bash
python vllm_cli.py add dev-model microsoft/DialoGPT-small --port 8004 --priority 5 --gpu-memory 0.1
python vllm_cli.py start dev-model
```

### Resource Monitoring
```bash
# Watch resource usage
watch -n 2 'python vllm_cli.py status'
```

## 🚀 Tab Completion

Enable smart autocompletion:

```bash
# Enable for current session
source .completion
```

**Usage (SPACE required before TAB):**
- `./vllm [SPACE][TAB]` → Shows: `cli test legacy`
- `./vllm cli [SPACE][TAB]` → Shows: `add list start stop remove status`
- `./vllm cli add model [TAB]` → Suggests popular HuggingFace models
- `--port [TAB]` → Suggests port numbers (8001, 8002, etc.)

## 🗂️ Legacy Files

The old simple manager is still available:
- `vllm-manager.sh` - Original simple shell-based manager
- Use for quick single-model setups

---

The terminal manager automatically handles resource conflicts, ensuring your high-priority models always have the resources they need! 🚀