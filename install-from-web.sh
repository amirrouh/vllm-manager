#!/bin/bash

# VLLM Manager Web Installer
# This script is meant to be hosted and downloaded with curl

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Installation paths
INSTALL_DIR="/opt/vllm-manager"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="$HOME/.vllm-manager"
REPO_URL="https://github.com/amirrouh/vllm-manager"

# Fancy banner
echo -e "${BLUE}${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         🚀 VLLM Manager Installer                         ║"
echo "║                  Single-Command Installation from Web                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}❌ Please don't run this script as root${NC}"
   echo "Run it as a regular user. The script will ask for password when needed."
   exit 1
fi

# Function to print status messages
print_status() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_step() {
    echo -e "${BOLD}🔧 $1${NC}"
}

# Check prerequisites
print_status "Checking system requirements..."

# Check for uv
if ! command -v uv &> /dev/null; then
    print_status "Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    print_success "✓ uv installed"
else
    print_success "✓ uv already installed"
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_success "✓ Python $PYTHON_VERSION found"

# Check system dependencies
print_step "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y build-essential python3-dev python3-pip curl git
print_success "✓ System dependencies installed"

# Create directories
print_step "Creating installation directories..."
sudo mkdir -p "$INSTALL_DIR"
sudo mkdir -p "$CONFIG_DIR"
sudo mkdir -p "$CONFIG_DIR/logs"
sudo chown -R $USER:$USER "$CONFIG_DIR"
print_success "✓ Directories created"

# Clone or download the repository
print_step "Downloading VLLM Manager..."
cd /tmp

if [ -d "vllm-manager" ]; then
    cd vllm-manager
    git pull
else
    git clone "$REPO_URL"
    cd vllm-manager
fi

# Copy files to installation directory
print_step "Installing files..."
sudo cp -r * "$INSTALL_DIR/"
cd "$INSTALL_DIR"
print_success "✓ Files installed"

# Setup uv environment
print_step "Setting up uv environment..."
uv init --app
uv add python>=3.8
uv add -r requirements.txt
uv add vllm  # This will install vllm in the uv environment
print_success "✓ uv environment configured"

# Create system-wide vm command
print_step "Creating vm command..."
sudo tee "$BIN_DIR/vm" > /dev/null << 'EOF'
#!/bin/bash
# VLLM Manager system-wide wrapper

INSTALL_DIR="/opt/vllm-manager"
CONFIG_DIR="$HOME/.vllm-manager"

# Ensure config directory exists
mkdir -p "$CONFIG_DIR"
mkdir -p "$CONFIG_DIR/logs"

# Change to installation directory
cd "$INSTALL_DIR"

# Handle special commands
if [[ "$1" == "help" ]] || [[ "$1" == "--help" ]] || [[ -z "$1" ]]; then
    echo "🚀 VLLM Manager - Modern Terminal Interface"
    echo ""
    echo "Usage: vm [command]"
    echo ""
    echo "Commands:"
    echo "  gui                 Launch the terminal interface (default)"
    echo "  add <name> <id>    Add a new model"
    echo "  list                List all models"
    echo "  start <name>        Start a model"
    echo "  stop <name>         Stop a model"
    echo "  remove <name>       Remove a model"
    echo "  status              Show system status"
    echo "  cleanup             Clean GPU memory"
    echo "  help                Show this help"
    echo ""
    echo "Examples:"
    echo "  vm gui              Launch interface"
    echo "  vm add mistral mistralai/Mistral-7B-Instruct-v0.2"
    echo "  vm start mistral"
    echo ""
    echo "Configuration directory: $CONFIG_DIR"
    exit 0
fi

# Use uv to run the application
exec uv run python vllm_manager.py "$@"
EOF

sudo chmod +x "$BIN_DIR/vm"
print_success "✓ vm command installed globally"

# Setup auto-completion
print_step "Setting up bash completion..."
sudo tee /etc/bash_completion.d/vm > /dev/null << 'EOF'
_vm_completion() {
    local cur prev words cword
    _init_completion || return

    case "${prev}" in
        vm)
            COMPREPLY=( $(compgen -W "gui add list start stop remove status cleanup help" -- "${cur}") )
            return 0
            ;;
        add)
            return 0
            ;;
        start|stop|remove)
            return 0
            ;;
        *)
            ;;
    esac
}

complete -F _vm_completion vm
EOF
print_success "✓ Bash completion installed"

# Final verification
print_step "Verifying installation..."
if command -v vm &> /dev/null; then
    print_success "✓ vm command is working"
else
    print_warning "⚠ vm command not found in PATH. You may need to restart your terminal."
fi

# Cleanup
rm -rf /tmp/vllm-manager

# Final success message
echo ""
echo -e "${GREEN}${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                          🎉 Installation Complete!                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo -e "${BOLD}You can now use VLLM Manager from anywhere:${NC}"
echo ""
echo -e "  ${BLUE}vm gui${NC}         - Launch the beautiful terminal interface"
echo -e "  ${BLUE}vm help${NC}        - Show all available commands"
echo -e "  ${BLUE}vm status${NC}      - Check system status"
echo ""
echo -e "${BOLD}What makes this special:${NC}"
echo -e "  🚀 ${GREEN}Zero setup${NC} - Everything is installed automatically"
echo -e "  📦 ${GREEN}uv-powered${NC} - Fast, modern Python package management"
echo -e "  🌍 ${GREEN}Global access${NC} - Use 'vm' from any directory"
echo -e "  🔄 ${GREEN}Self-contained${NC} - vLLM installs automatically on first run"
echo ""
echo -e "${BOLD}Configuration:${NC}"
echo "  Config directory: $CONFIG_DIR"
echo "  Installation: $INSTALL_DIR"
echo ""
echo -e "${YELLOW}💡 Pro tips:${NC}"
echo "  • Your first run will automatically install vLLM using uv"
echo "  • Use 'vm gui' to launch the interface anytime"
echo "  • Tab completion is available for commands"
echo ""
echo -e "${BOLD}🚀 Ready to run LLMs? Try: ${BLUE}vm gui${NC}"
echo ""

# Ask if user wants to run it now
read -p "Would you like to launch VLLM Manager now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Launching VLLM Manager..."
    vm gui
fi