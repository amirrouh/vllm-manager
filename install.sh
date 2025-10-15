#!/bin/bash

# VLLM Manager Installer
# Single-command system-wide installation like Homebrew

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
echo "║                   Single-Command Installation like Homebrew                ║"
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

# Update the main script to use uv
print_step "Configuring VLLM Manager for uv..."
cat > vllm_manager_uv.py << 'EOF'
#!/usr/bin/env python3
"""
VLLM Manager - Modern Terminal Interface
uv-enabled version
"""

import subprocess
import sys
from pathlib import Path

def run_with_uv():
    """Run vllm_manager.py using uv"""
    install_dir = Path(__file__).parent
    cmd = ['uv', 'run', 'python', 'vllm_manager.py'] + sys.argv[1:]

    # Handle the gui command specially
    if len(sys.argv) <= 1 or sys.argv[1] == 'gui':
        cmd = ['uv', 'run', 'python', 'vllm_manager.py', 'gui']

    os.execvp('uv', cmd)

if __name__ == "__main__":
    run_with_uv()
EOF

# Create system-wide vm command
print_step "Creating vm command..."
sudo tee "$BIN_DIR/vm" > /dev/null << EOF
#!/bin/bash
# VLLM Manager system-wide wrapper

INSTALL_DIR="/opt/vllm-manager"
CONFIG_DIR="\$HOME/.vllm-manager"

# Ensure config directory exists
mkdir -p "\$CONFIG_DIR"
mkdir -p "\$CONFIG_DIR/logs"

# Change to installation directory
cd "\$INSTALL_DIR"

# Handle special commands
if [[ "\$1" == "help" ]] || [[ "\$1" == "--help" ]] || [[ -z "\$1" ]]; then
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
    echo "  uninstall           Uninstall VLLM Manager"
    echo "  help                Show this help"
    echo ""
    echo "Examples:"
    echo "  vm gui              Launch interface"
    echo "  vm add mistral mistralai/Mistral-7B-Instruct-v0.2"
    echo "  vm start mistral"
    echo "  vm uninstall        Remove VLLM Manager"
    echo ""
    echo "Configuration directory: \$CONFIG_DIR"
    exit 0
fi

# Handle uninstall command
if [[ "\$1" == "uninstall" ]]; then
    echo "🔥 VLLM Manager Uninstaller"
    echo "================================="
    echo ""
    echo "This will completely remove VLLM Manager from your system."
    echo ""
    read -p "Are you sure you want to uninstall? [y/N] " -n 1 -r
    echo
    if [[ ! \$REPLY =~ ^[Yy]$ ]]; then
        echo "Uninstall cancelled."
        exit 0
    fi

    read -p "Also remove user data (models, logs, config)? [y/N] " -n 1 -r
    echo
    if [[ \$REPLY =~ ^[Yy]$ ]]; then
        REMOVE_DATA=true
    else
        REMOVE_DATA=false
    fi

    echo "Uninstalling VLLM Manager..."

    # Stop processes
    pkill -f "vllm_manager.py" 2>/dev/null || true
    pkill -f "vllm entrypoints" 2>/dev/null || true

    # Remove files
    sudo rm -rf "$INSTALL_DIR"
    sudo rm -f "$BIN_DIR/vm"
    sudo rm -f "/etc/bash_completion.d/vm"
    sudo rm -f "/usr/share/applications/vllm-manager.desktop"
    sudo rm -f "/usr/local/share/man/man1/vm.1.gz"

    if [[ "$REMOVE_DATA" = true ]]; then
        rm -rf "$CONFIG_DIR"
    fi

    echo ""
    echo "✅ VLLM Manager has been uninstalled!"

    if [[ "$REMOVE_DATA" = false ]]; then
        echo "User data preserved in: $CONFIG_DIR"
    fi

    echo "You may need to restart your terminal."
    exit 0
fi

# Use uv to run the application
exec uv run python vllm_manager.py "\$@"
EOF

sudo chmod +x "$BIN_DIR/vm"
print_success "✓ vm command installed globally"

# Create desktop entry for GUI
print_step "Creating desktop entry..."
sudo tee /usr/share/applications/vllm-manager.desktop > /dev/null << EOF
[Desktop Entry]
Name=VLLM Manager
Comment=Modern terminal interface for vLLM model management
Exec=vm gui
Icon=terminal
Terminal=true
Type=Application
Categories=Development;System;
StartupNotify=true
EOF
print_success "✓ Desktop entry created"

# Create man page
print_step "Creating documentation..."
sudo tee /usr/local/share/man/man1/vm.1 > /dev/null << 'EOF'
.TH VM 1 "VLLM Manager" "User Commands"
.SH NAME
vm \- VLLM Manager - Modern Terminal Interface for vLLM Model Management
.SH SYNOPSIS
.B vm
[\fIcommand\fR] [\fIoptions\fR]
.SH DESCRIPTION
VLLM Manager is a modern terminal-based interface for managing multiple vLLM models with real-time monitoring and intelligent resource allocation.
.SH COMMANDS
.TP
\fBgui\fR
Launch the terminal interface (default command)
.TP
\fBadd\fR \fIname\fR \fIhuggingface_id\fR
Add a new model with the specified name and HuggingFace ID
.TP
\fBlist\fR
List all configured models
.TP
\fBstart\fR \fIname\fR
Start a model by name
.TP
\fBstop\fR \fIname\fR
Stop a running model
.TP
\fBremove\fR \fIname\fR
Remove a model configuration
.TP
\fBstatus\fR
Show system status and GPU information
.TP
\fBcleanup\fR
Clean GPU memory by stopping low-priority models
.TP
\fBhelp\fR
Show this help message
.SH EXAMPLES
Launch the terminal interface:
.PP
.RS 4
vm gui
.RE
.PP
Add a Mistral model:
.PP
.RS 4
vm add mistral mistralai/Mistral-7B-Instruct-v0.2
.RE
.PP
Start a model:
.PP
.RS 4
vm start mistral
.RE
.SH FILES
.TP
.I ~/.vllm-manager/
Configuration directory for VLLM Manager
.TP
.I /opt/vllm-manager/
Installation directory
.SH AUTHOR
VLLM Manager contributors
.SH "SEE ALSO"
The full documentation is available at: https://github.com/amirrouh/vllm-manager
EOF

sudo gzip -f /usr/local/share/man/man1/vm.1
print_success "✓ Man page installed"

# Setup auto-completion
print_step "Setting up bash completion..."
sudo tee /etc/bash_completion.d/vm > /dev/null << 'EOF'
_vm_completion() {
    local cur prev words cword
    _init_completion || return

    case "${prev}" in
        vm)
            COMPREPLY=( $(compgen -W "gui add list start stop remove status cleanup uninstall help" -- "${cur}") )
            return 0
            ;;
        add)
            return 0
            ;;
        start|stop|remove)
            # This could be enhanced to read actual model names
            return 0
            ;;
        *)
            ;;
    esac
}

complete -F _vm_completion vm
EOF
print_success "✓ Bash completion installed"

# Set up systemd user service for auto-restart (optional)
print_step "Setting up systemd user service..."
mkdir -p "$CONFIG_DIR/systemd-user"
tee "$CONFIG_DIR/systemd-user/vllm-manager.service" > /dev/null << EOF
[Unit]
Description=VLLM Manager Service
After=network.target

[Service]
Type=simple
ExecStart=$BIN_DIR/vm gui
Restart=on-failure
RestartSec=5
Environment=HOME=$HOME
WorkingDirectory=$INSTALL_DIR

[Install]
WantedBy=default.target
EOF

print_success "✓ Systemd service template created"

# Final verification
print_step "Verifying installation..."
if command -v vm &> /dev/null; then
    VM_VERSION=$(vm --version 2>/dev/null || echo "installed")
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
echo -e "  📚 ${GREEN}Complete${NC} - Documentation, completion, desktop entry"
echo ""
echo -e "${BOLD}Configuration:${NC}"
echo "  Config directory: $CONFIG_DIR"
echo "  Installation: $INSTALL_DIR"
echo ""
echo -e "${YELLOW}💡 Pro tips:${NC}"
echo "  • Your first run will automatically install vLLM using uv"
echo "  • Use 'vm gui' to launch the interface anytime"
echo "  • Tab completion is available for commands"
echo "  • Check 'man vm' for detailed documentation"
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