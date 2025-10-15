# 📖 VLLM Manager Documentation

## Getting Started

### First Run

The first time you run `vm gui`, VLLM Manager will automatically:
- Create a virtual environment
- Download and install vLLM (5-15 minutes)
- Configure everything for you
- Launch the interface

### Navigation

```
↑/↓        • Select models
ENTER      • Start/Stop models
A          • Add new models
S          • Configure settings
D          • Delete models
C          • Clean GPU memory
H          • Show help
Q          • Quit
```

## Features

### Smart Resource Management
- **Auto-cleanup** - Frees GPU memory when needed
- **Priority system** - Important models get resources first
- **Health monitoring** - Restarts crashed models automatically
- **Port management** - No conflicts, ever

### Beautiful Interface
- **Modern design** - Clean, professional terminal UI
- **Real-time updates** - Live stats and monitoring
- **Intuitive dialogs** - No need to remember commands
- **Help system** - Press `H` anytime

### Power User Features
- **Multi-model support** - Run different models simultaneously
- **GPU optimization** - Smart memory allocation
- **Error recovery** - Graceful handling of failures
- **Configuration persistence** - Settings saved automatically

## Popular Models

Copy-paste these when adding models:

- **Mistral 7B**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Llama 3 8B**: `meta-llama/Llama-3-8B-Instruct`
- **Llama 2 7B**: `meta-llama/Llama-2-7b-chat-hf`
- **Mixtral 8x7B**: `mistralai/Mixtral-8x7B-Instruct-v0.1`

## Hugging Face Access

For gated models (like Llama), create a `.env` file:

```env
# Get your token: https://huggingface.co/settings/tokens
HF_TOKEN=your_token_here
```

## System Requirements

- **Linux** with NVIDIA GPU
- **Python 3.8+**
- **8GB+ RAM** recommended
- **10GB+ storage** for models

## Pro Tips

- **First models** start with Mistral 7B - it's fast and lightweight
- **GPU memory** start with 0.3 and adjust based on performance
- **Ports** are automatically assigned, but you can customize
- **Priority** set higher (1-2) for models you use frequently
- **Cleanup** press `C` if models seem slow or unresponsive

## Uninstalling

### Easy Uninstall

```bash
vm uninstall
```

### Manual Uninstall

```bash
# Remove installation
rm -rf ~/.vllm-manager
rm -f ~/.local/bin/vm

# Remove bash completion
rm -f ~/.bash_completion.d/vm
```

## Troubleshooting

### Installation Issues

**Permission denied**: The installer now runs as user, not root

**PATH issues**: Make sure `~/.local/bin` is in your PATH:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### vLLM Installation

**Slow installation**: vLLM is a large package, first-time installation takes 5-15 minutes

**Memory issues**: Ensure you have enough disk space and RAM

### Model Issues

**Download failures**: Check your internet connection and Hugging Face access

**GPU memory**: Adjust memory allocation in settings or free up GPU memory with `C`

---

*MIT License • Contributions welcome • Star us on GitHub!* ⭐