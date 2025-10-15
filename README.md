# VLLM Terminal Manager

A complete terminal-based interface for managing multiple VLLM models with smart GPU resource allocation. **All operations are contained within the UI - no command line needed!**

## Terminal Interface

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          VLLM Terminal Manager                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🖥️  GPU Status: NVIDIA RTX 4090 [8192MB/24576MB used] [33% utilization]   ║
║  🌡️  Temperature: 42°C                                                       ║
║                                                                              ║
║  🤖 MODELS:                                                                  ║
║                                                                              ║
║  🟢 llama3-8b    [PID: 1234] [Port: 8001] [Priority: 1]                    ║
║       Status: Running | GPU: 2048MB | CPU: 2.3% | Uptime: 2h 15m           ║
║                                                                              ║
║  🟡 mistral-7b   [PID: ----] [Port: 8002] [Priority: 2]                    ║
║       Status: Starting...                                                   ║
║                                                                              ║
║  ⚫ codellama     [PID: ----] [Port: 8003] [Priority: 4]                    ║
║       Status: Stopped                                                       ║
║                                                                              ║
║  🎮 CONTROLS:                                                                ║
║    ↑/↓ Navigate | Enter Start/Stop | A Add | S Settings | D Delete          ║
║    C Cleanup | K Kill Process | H Help | Q Quit                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Quick Start

### Launch the Terminal UI
```bash
./vm gui
```

**That's it! Everything is managed through the interface:**

- **Add Models** - Press `A` to open the add model dialog
- **Configure Settings** - Press `S` to adjust GPU memory, priority, etc.
- **Start/Stop Models** - Navigate with arrow keys and press Enter
- **Remove Models** - Press `D` with confirmation dialog
- **GPU Cleanup** - Press `C` to free memory with confirmation
- **Kill Processes** - Press `K` to force-stop unresponsive models

## Features

- **Complete UI Management** - All operations through terminal interface, no CLI needed
- **Real-time GPU monitoring** - Track memory, utilization, and temperature
- **Interactive dialogs** - Add models, configure settings, confirm deletions
- **Priority system** - High-priority models get resources automatically
- **Multi-model support** - Run several models simultaneously
- **Smart cleanup** - Free memory by stopping low-priority models
- **Process management** - Kill unresponsive processes safely

## Complete UI Controls

| Key | Action |
|-----|--------|
| ↑/↓ | Navigate models |
| Enter/Space | Start/Stop selected model |
| A | Add new model (interactive dialog) |
| S | Model settings (GPU memory, priority, etc.) |
| D | Delete model (with confirmation) |
| C | GPU cleanup (with confirmation) |
| K | Kill model process (force stop) |
| H | Show help |
| Q | Quit application |

## Interactive Dialogs

The UI now includes full-featured dialogs for all operations:

- **Add Model Dialog** - Configure name, HuggingFace ID, port, priority, GPU memory
- **Settings Dialog** - Modify GPU memory, priority levels, model parameters
- **Confirmation Dialogs** - Safe deletion and cleanup with process information
- **Help System** - Complete keyboard reference and usage guide

## Priority Levels

- **Priority 1-2**: Protected (never auto-killed)
- **Priority 3**: Medium priority
- **Priority 4-5**: Low priority (first to be killed)

## Status Indicators

- 🟢 Running
- 🟡 Starting
- 🔴 Error
- ⚫ Stopped

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA
- vLLM installed

---

**No Command Line Required!** The VLLM Manager is now a complete self-contained terminal interface. All model management operations are available through interactive dialogs - just launch `./vm gui` and manage everything from the comfort of your terminal!

*The terminal manager automatically handles resource conflicts to keep your high-priority models running!*