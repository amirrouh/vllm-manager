#!/bin/bash

# VLLM Manager Web Installer
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

INSTALL_DIR="$HOME/.vllm-manager"
BIN_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.vllm-manager"
REPO_URL="https://github.com/amirrouh/vllm-manager"

echo -e "${BLUE}${BOLD}VLLM Manager Installer${NC}"
echo -e "${BLUE}Single-Command Installation from Web${NC}"

if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}❌ Please don't run this script as root${NC}"
   exit 1
fi

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_step() {
    echo -e "${BLUE}🔧 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

if command_exists uv; then
    print_success "✓ uv already installed"
else
    print_step "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    print_success "✓ uv installed"
fi

if command_exists python3 && python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_success "✓ Python 3.8+ found"
else
    echo -e "${RED}❌ Python 3.8+ is required${NC}"
    exit 1
fi

if command_exists git; then
    print_success "✓ git already installed"
else
    print_step "Installing git..."
    sudo apt-get update -qq
    sudo apt-get install -y git
    print_success "✓ git installed"
fi

print_step "Creating installation directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$CONFIG_DIR/logs"
print_success "✓ Directories created"

print_step "Downloading VLLM Manager..."
cd /tmp
rm -rf vllm-manager
git clone "$REPO_URL"
cd vllm-manager

print_step "Installing files..."
echo -ne "${YELLOW}Installing files...${NC}"
for i in {1..5}; do
    echo -ne "."
    sleep 0.3
done
echo ""

mkdir -p "$INSTALL_DIR"
cp -r * "$INSTALL_DIR/"
cd "$INSTALL_DIR"
print_success "✓ Files installed"

print_step "Setting up uv environment..."
echo -ne "${YELLOW}Setting up uv environment...${NC}"
for i in {1..8}; do
    echo -ne "."
    sleep 0.3
done
echo ""

cat > pyproject.toml << 'INNEREOF'
[project]
name = "vllm-manager"
version = "0.1.0"
description = "Zero-Setup vLLM Model Management"
requires-python = ">=3.8"
dependencies = [
    "vllm",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
INNEREOF

uv add vllm
print_success "✓ uv environment configured"

print_step "Creating vm command..."
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/vm" << 'INNEREOF'
#!/bin/bash
INSTALL_DIR="$HOME/.vllm-manager"
CONFIG_DIR="$HOME/.vllm-manager"
mkdir -p "$CONFIG_DIR"
mkdir -p "$CONFIG_DIR/logs"
cd "$INSTALL_DIR"
if [[ "$1" == "help" ]] || [[ "$1" == "--help" ]] || [[ -z "$1" ]]; then
    echo "🚀 VLLM Manager"
    echo ""
    echo "Usage: vm [command]"
    echo ""
    echo "Commands:"
    echo "  gui                 Launch the interface"
    echo "  add <name> <id>    Add a new model"
    echo "  list                List all models"
    echo "  start <name>        Start a model"
    echo "  stop <name>         Stop a model"
    echo "  remove <name>       Remove a model"
    echo "  status              Show system status"
    echo "  cleanup             Clean GPU memory"
    echo "  uninstall           Uninstall VLLM Manager"
    echo "  help                Show this help"
    exit 0
fi
exec uv run python vllm_manager.py "$@"
INNEREOF

chmod +x "$BIN_DIR/vm"
print_success "✓ vm command installed for user"

print_step "Setting up bash completion..."
mkdir -p "$HOME/.bash_completion.d"
cat > "$HOME/.bash_completion.d/vm" << 'INNEREOF'
_vm_completion() {
    local cur prev words cword
    _init_completion || return
    case "${prev}" in
        vm)
            COMPREPLY=( $(compgen -W "gui add list start stop remove status cleanup help" -- "${cur}") )
            return 0
            ;;
    esac
}
complete -F _vm_completion vm
INNEREOF

if ! grep -q "source.*bash_completion.d/vm" "$HOME/.bashrc"; then
    echo 'source ~/.bash_completion.d/vm' >> "$HOME/.bashrc"
fi
print_success "✓ Bash completion installed"

print_step "Verifying installation..."
if command -v vm &> /dev/null; then
    print_success "✓ vm command is working"
else
    echo -e "${RED}❌ vm command not found in PATH${NC}"
    echo -e "${YELLOW}Make sure $HOME/.local/bin is in your PATH${NC}"
fi

rm -rf /tmp/vllm-manager

echo ""
echo -e "${GREEN}${BOLD}🎉 Installation Complete!${NC}"
echo ""
echo -e "${BOLD}You can now use VLLM Manager:${NC}"
echo ""
echo -e "  ${BLUE}vm gui${NC}         - Launch the interface"
echo -e "  ${BLUE}vm help${NC}        - Show commands"
echo ""
echo -e "${YELLOW}💡 Make sure $HOME/.local/bin is in your PATH${NC}"
echo -e "${BOLD}🚀 Ready to run LLMs? Try: ${BLUE}vm gui${NC}"
echo ""

read -p "Would you like to launch VLLM Manager now? [y/N] " -n 1 -r
printf "\n"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Launching VLLM Manager..."
    vm gui
fi
