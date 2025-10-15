# 🚀 VLLM Manager

**Zero-Setup vLLM Model Management**
*A beautiful terminal interface that just works*

## ✨ Why VLLM Manager?

🔥 **Zero Dependencies** - No complex setup, vLLM installs automatically
💫 **Beautiful UI** - Modern terminal interface with real-time monitoring
🎯 **One-Click Everything** - Add models, adjust settings, manage resources
⚡ **Smart Resource Management** - Automatic GPU optimization and cleanup
🔄 **Complete Control** - Everything happens in the interface, no CLI needed

## 🚀 Getting Started

### **The Ultimate One-Command Installation**

```bash
curl -fsSL https://raw.githubusercontent.com/amirrouh/vllm-manager/main/install-from-web.sh | bash
```

**That's literally it!** One command and everything is installed globally. 🎉

### **What this magical command does:**
- ✅ Installs uv (lightning-fast Python package manager)
- ✅ Downloads VLLM Manager from GitHub
- ✅ Sets up everything in `/opt/vllm-manager`
- ✅ Creates the `vm` command globally available
- ✅ Installs bash completion
- ✅ Sets up man pages and desktop entries
- ✅ Configures everything for uv-based package management

### **After installation, use it anywhere:**

```bash
vm gui                    # Launch the interface (from any directory!)
vm help                   # See all commands
vm status                 # Check system status
vm add mistral mistralai/Mistral-7B-Instruct-v0.2  # Add a model
```

### **Traditional Installation (if you prefer):**

```bash
# Clone and install manually
git clone https://github.com/amirrouh/vllm-manager.git
cd vllm-manager
./install.sh
```

## 🔧 Uninstalling VLLM Manager

### **The Easiest Way - Built-in Uninstall**

```bash
vm uninstall
```

**That's it!** The `vm` command includes its own uninstaller. Just type `vm uninstall` and follow the prompts.

### **What uninstallation removes:**
- ✅ Installation directory (`/opt/vllm-manager`)
- ✅ Global `vm` command from `/usr/local/bin`
- ✅ Bash completion files
- ✅ Desktop entries
- ✅ Man pages
- ✅ System services
- 🤔 User data (optional - you'll be asked)

### **Other Ways to Uninstall:**

```bash
# If you can access the vm command:
vm uninstall

# If vm command is broken, use the web uninstaller:
curl -fsSL https://raw.githubusercontent.com/amirrouh/vllm-manager/main/uninstall-from-web.sh | bash

# Manual uninstall (last resort):
sudo rm -rf /opt/vllm-manager
sudo rm -f /usr/local/bin/vm
rm -rf ~/.vllm-manager
```

### **After uninstallation:**
- The `vm` command will no longer be available
- You may need to restart your terminal for shell completion to update
- Your models and logs will be preserved unless you choose to remove them

## 🎮 How to Use

### **First Run Magic**
The app detects missing vLLM and automatically:
- Creates a local virtual environment
- Downloads and installs vLLM (5-15 minutes)
- Configures everything for you
- Launches the beautiful interface

### **Navigate Like a Pro**
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

### **What You'll See**
- 📊 **Live GPU Stats** - Memory usage, temperature, utilization
- 🎮 **Model Dashboard** - Status, ports, resource usage
- ⚡ **Smart Controls** - Intuitive dialogs for everything
- 🛡️ **Health Monitoring** - Automatic restarts, error handling

## 💫 Popular Models You Can Run

**Just copy-paste these when adding models:**

- **Mistral 7B**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Llama 3 8B**: `meta-llama/Llama-3-8B-Instruct`
- **Llama 2 7B**: `meta-llama/Llama-2-7b-chat-hf`
- **Mixtral 8x7B**: `mistralai/Mixtral-8x7B-Instruct-v0.1`

## 🔧 Optional: Hugging Face Access

For gated models (like Llama), create a `.env` file:

```env
# Get your token: https://huggingface.co/settings/tokens
HF_TOKEN=your_token_here
```

## 🎨 Features That Make You Smile

### **Smart Resource Management**
- **Auto-cleanup** - Frees GPU memory when needed
- **Priority system** - Important models get resources first
- **Health monitoring** - Restarts crashed models automatically
- **Port management** - No conflicts, ever

### **Beautiful Interface**
- **Modern design** - Clean, professional terminal UI
- **Real-time updates** - Live stats and monitoring
- **Intuitive dialogs** - No need to remember commands
- **Help system** - Press `H` anytime

### **Power User Features**
- **Multi-model support** - Run different models simultaneously
- **GPU optimization** - Smart memory allocation
- **Error recovery** - Graceful handling of failures
- **Configuration persistence** - Settings saved automatically

## 🎯 What Makes It Special

- **One-command install** - Like Homebrew: `curl | bash`
- **Globally available** - Use `vm` from any directory
- **uv-powered** - Lightning-fast Python package management
- **Zero configuration** - Everything just works out of the box
- **Beautiful interface** - No CLI required, everything in the UI
- **Self-contained** - Installs to `/opt/vllm-manager`, no conflicts

## 💡 Pro Tips

- **First models** start with Mistral 7B - it's fast and lightweight
- **GPU memory** start with 0.3 and adjust based on performance
- **Ports** are automatically assigned, but you can customize
- **Priority** set higher (1-2) for models you use frequently
- **Cleanup** press `C` if models seem slow or unresponsive

## 🖥️ System Requirements

- **Linux** with NVIDIA GPU
- **Python 3.8+**
- **8GB+ RAM** recommended
- **10GB+ storage** for models

## 🎪 What People Love

> *"Finally! A vLLM interface that doesn't require a PhD to use."*
> *"The auto-installation saved me hours of setup headaches."*
> *"Beautiful UI and everything just works. This is how LLM management should be."*

---

**Built with ❤️ for humans who want to focus on LLMs, not configuration**

*MIT License • Contributions welcome • Star us on GitHub!* ⭐