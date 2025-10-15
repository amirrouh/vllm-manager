#!/bin/bash

# VLLM Manager Uninstaller
# Completely removes VLLM Manager from your system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Installation paths (same as installer)
INSTALL_DIR="/opt/vllm-manager"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="$HOME/.vllm-manager"

# Fancy banner
echo -e "${RED}${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                       🔥 VLLM Manager Uninstaller                         ║"
echo "║                  Completely remove VLLM Manager from your system             ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${YELLOW}⚠️  Running as root. Will uninstall for all users.${NC}"
   ROOT_UNINSTALL=true
else
   echo -e "${BLUE}ℹ️  Running as user. Will uninstall for current user.${NC}"
   ROOT_UNINSTALL=false
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

# Warning and confirmation
echo ""
print_warning "This will completely remove VLLM Manager from your system, including:"
echo "  • Installation directory: $INSTALL_DIR"
echo "  • User configuration: $CONFIG_DIR"
echo "  • Global command: $BIN_DIR/vm"
echo "  • Bash completion"
echo "  • Desktop entry"
echo "  • Man pages"
echo ""

read -p "Are you sure you want to uninstall VLLM Manager? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstall cancelled."
    exit 0
fi

# Ask about user data
read -p "Also remove user data (models, logs, configuration)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    REMOVE_USER_DATA=true
else
    REMOVE_USER_DATA=false
fi

# Stop any running processes
print_step "Stopping any running VLLM Manager processes..."
pkill -f "vllm_manager.py" 2>/dev/null || true
pkill -f "vllm entrypoints" 2>/dev/null || true
print_success "✓ Processes stopped"

# Remove installation directory
if [[ -d "$INSTALL_DIR" ]]; then
    print_step "Removing installation directory..."
    if [[ "$ROOT_UNINSTALL" = true ]]; then
        sudo rm -rf "$INSTALL_DIR"
    else
        sudo rm -rf "$INSTALL_DIR" 2>/dev/null || print_warning "Could not remove $INSTALL_DIR (needs sudo)"
    fi
    print_success "✓ Installation directory removed"
else
    print_warning "Installation directory not found: $INSTALL_DIR"
fi

# Remove global command
if [[ -f "$BIN_DIR/vm" ]]; then
    print_step "Removing global vm command..."
    if [[ "$ROOT_UNINSTALL" = true ]]; then
        sudo rm -f "$BIN_DIR/vm"
    else
        sudo rm -f "$BIN_DIR/vm" 2>/dev/null || print_warning "Could not remove $BIN_DIR/vm (needs sudo)"
    fi
    print_success "✓ Global command removed"
else
    print_warning "Global command not found: $BIN_DIR/vm"
fi

# Remove bash completion
if [[ -f "/etc/bash_completion.d/vm" ]]; then
    print_step "Removing bash completion..."
    if [[ "$ROOT_UNINSTALL" = true ]]; then
        sudo rm -f "/etc/bash_completion.d/vm"
    else
        sudo rm -f "/etc/bash_completion.d/vm" 2>/dev/null || print_warning "Could not remove bash completion (needs sudo)"
    fi
    print_success "✓ Bash completion removed"
fi

# Remove desktop entry
if [[ -f "/usr/share/applications/vllm-manager.desktop" ]]; then
    print_step "Removing desktop entry..."
    if [[ "$ROOT_UNINSTALL" = true ]]; then
        sudo rm -f "/usr/share/applications/vllm-manager.desktop"
    else
        sudo rm -f "/usr/share/applications/vllm-manager.desktop" 2>/dev/null || print_warning "Could not remove desktop entry (needs sudo)"
    fi
    print_success "✓ Desktop entry removed"
fi

# Remove man page
if [[ -f "/usr/local/share/man/man1/vm.1.gz" ]]; then
    print_step "Removing man page..."
    if [[ "$ROOT_UNINSTALL" = true ]]; then
        sudo rm -f "/usr/local/share/man/man1/vm.1.gz"
    else
        sudo rm -f "/usr/local/share/man/man1/vm.1.gz" 2>/dev/null || print_warning "Could not remove man page (needs sudo)"
    fi
    print_success "✓ Man page removed"
fi

# Remove user data if requested
if [[ "$REMOVE_USER_DATA" = true ]]; then
    if [[ -d "$CONFIG_DIR" ]]; then
        print_step "Removing user configuration and data..."
        rm -rf "$CONFIG_DIR"
        print_success "✓ User data removed"
    fi
else
    if [[ -d "$CONFIG_DIR" ]]; then
        print_warning "User data preserved in: $CONFIG_DIR"
        print_warning "To remove manually: rm -rf $CONFIG_DIR"
    fi
fi

# Clear uv cache if requested
if [[ "$REMOVE_USER_DATA" = true ]]; then
    print_step "Cleaning up uv cache..."
    uv cache clean 2>/dev/null || true
    print_success "✓ UV cache cleaned"
fi

# Remove any remaining vllm processes from other users
if [[ "$ROOT_UNINSTALL" = true ]]; then
    print_step "Cleaning up any remaining vllm processes..."
    pkill -9 -f "vllm" 2>/dev/null || true
    print_success "✓ Process cleanup completed"
fi

# Update shell cache
print_step "Updating shell cache..."
hash -r 2>/dev/null || true

# Final verification
print_step "Verifying uninstallation..."

UNINSTALL_COMPLETE=true

# Check if vm command still exists
if command -v vm &> /dev/null; then
    print_error "❌ vm command still found. You may need to restart your terminal."
    UNINSTALL_COMPLETE=false
fi

# Check if installation directory still exists
if [[ -d "$INSTALL_DIR" ]]; then
    print_error "❌ Installation directory still exists: $INSTALL_DIR"
    UNINSTALL_COMPLETE=false
fi

# Final message
echo ""
if [[ "$UNINSTALL_COMPLETE" = true ]]; then
    echo -e "${GREEN}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                          🎉 Uninstall Complete!                          ║"
    echo "║                     VLLM Manager has been removed                       ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo "✅ VLLM Manager has been completely uninstalled!"

    if [[ "$REMOVE_USER_DATA" = false ]]; then
        echo ""
        print_warning "User data was preserved:"
        echo "  Configuration: $CONFIG_DIR"
        echo "  To remove manually: rm -rf $CONFIG_DIR"
    fi

    echo ""
    echo "💡 You may need to restart your terminal for all changes to take effect."
    echo ""
    echo "👋 Thanks for using VLLM Manager!"

else
    echo -e "${YELLOW}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                        ⚠️  Partial Uninstall                             ║"
    echo "║                  Some components may still remain                          ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    print_error "Some components could not be removed."
    echo ""
    echo "You may need to:"
    echo "  • Run with sudo: sudo $0"
    echo "  • Restart your terminal and run again"
    echo "  • Manually remove remaining files"
    echo ""
    echo "Remaining files to check:"
    echo "  • $BIN_DIR/vm"
    echo "  • $INSTALL_DIR"
    echo "  • $CONFIG_DIR"
    echo "  • /etc/bash_completion.d/vm"
    echo "  • /usr/share/applications/vllm-manager.desktop"
fi