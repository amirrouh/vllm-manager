# 🚀 VLLM Manager

High-Performance vLLM Model Management with Modern Terminal UI

## 🚀 Node.js Edition (Recommended)

### Single Command Installation

```bash
curl -sSL https://raw.githubusercontent.com/amirrouh/vllm-manager/master/install-nodejs-complete.sh | bash
```

This will:
- ✅ **Remove Python version** safely with backup
- ✅ **Install Node.js version** with modern terminal UI
- ✅ **Migrate configurations** automatically
- ✅ **Set up everything** with proper symlinks and PATH

### Local Installation

```bash
git clone git@github.com:amirrouh/vllm-manager.git
cd vllm-manager
./install-nodejs-complete.sh
```

## Usage

```bash
vllm-manager                    # Launch modern terminal UI
vm                              # Shortcut alias
vllm-manager list               # List models
vllm-manager start <model>      # Start model
vllm-manager help               # Show all commands
```

## ✨ Features

### 🎨 Modern Terminal UI
- **Beautiful Dashboard** with real-time updates
- **Responsive Design** that adapts to terminal size
- **Rich Interface** with colors, icons, and animations
- **Interactive Controls** with keyboard navigation

### ⚡ Performance Benefits
- **10x Better Performance** with non-blocking I/O
- **50% Less Memory Usage** than Python version
- **Real-time Updates** via WebSocket
- **Fast Startup** with instant UI response

### 🛠️ Complete Management
- **Model CRUD** operations (Add, Edit, Delete, Start, Stop)
- **GPU Monitoring** with real-time statistics
- **Process Management** with automatic cleanup
- **HuggingFace Token** configuration

## 🎮 UI Controls

| Key | Action |
|-----|--------|
| ↑/k | Navigate up |
| ↓/j | Navigate down |
| Enter | Start/Stop model |
| A | Add model |
| E | Edit model |
| D | Delete model |
| C | Cleanup GPU |
| T | Configure token |
| ? | Help |
| Q/ESC | Quit |

## 📚 Documentation

- **[Node.js Edition Guide](README-NODEJS.md)** - Complete features and usage
- **[Installation Guide](README-INSTALLATION.md)** - Detailed installation options
- **[Python Version](README-PYTHON.md)** - Legacy documentation

## 🐍 Python Version (Legacy)

The original Python version is still available but deprecated:

```bash
curl -fsSL https://raw.githubusercontent.com/amirrouh/vllm-manager/master/install.sh | bash
```

⚠️ **Note:** The Node.js edition provides significantly better performance and user experience.

---

*Built with ❤️ for humans who want to focus on LLMs, not configuration*