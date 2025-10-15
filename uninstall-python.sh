#!/bin/bash

# VLLM Manager Python Version Uninstaller
# This script safely removes the Python VLLM Manager installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print() {
    echo -e "${2}$1${NC}"
}

print_bold() {
    echo -e "${BOLD}$1${NC}"
}

print_success() {
    print "$1" "$GREEN"
}

print_error() {
    print "$1" "$RED"
}

print_warning() {
    print "$1" "$YELLOW"
}

print_info() {
    print "$1" "$BLUE"
}

# ASCII Art
print_logo() {
    echo -e "${RED}"
    cat << 'EOF'
    ____              __    ____            _
   / __ \____  __  __/ /   / __ \____  ____(_)
  / /_/ / __ \/ / / / /   / /_/ / __ \/ __/ /
 / ____/ /_/ / /_/ / /___/ ____/ /_/ / /_/ /
/_/    \____/\__,_/_____/_/    \____/\__,_/

           PYTHON VERSION UNINSTALLER
EOF
    echo -e "${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "❌ Please do not run this uninstaller as root!"
    exit 1
fi

print_logo
print_bold "🗑️  VLLM Manager Python Version Uninstaller"
echo

# Confirmation dialog
print_warning "⚠️  This will remove the Python VLLM Manager from your system."
print_warning "⚠️  All your model configurations and logs will be REMOVED."
echo

read -p "$(echo -e ${YELLOW}"Are you sure you want to continue? (y/N): "${NC})" -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "❌ Uninstallation cancelled."
    exit 0
fi

print_info "🔍 Scanning for Python VLLM Manager installation..."

# Define potential installation locations
INSTALL_LOCATIONS=(
    "$HOME/.local/bin/vm"
    "$HOME/.local/bin/vllm-manager"
    "/usr/local/bin/vm"
    "/usr/local/bin/vllm-manager"
    "$HOME/apps/vllm-manager"
    "$HOME/vllm-manager"
    "$HOME/.vllm-manager"
    "$HOME/vllm-manager-python"
)

# Found locations
FOUND_LOCATIONS=()

# Check each location
for location in "${INSTALL_LOCATIONS[@]}"; do
    if [ -e "$location" ]; then
        FOUND_LOCATIONS+=("$location")
    fi
fi

# Also check for systemd services
SYSTEMD_SERVICES=(
    "$HOME/.config/systemd/user/vllm-manager.service"
    "/etc/systemd/system/vllm-manager.service"
)

for service in "${SYSTEMD_SERVICES[@]}"; do
    if [ -f "$service" ]; then
        FOUND_LOCATIONS+=("$service")
    fi
fi

# Show what will be removed
if [ ${#FOUND_LOCATIONS[@]} -eq 0 ]; then
    print_warning "⚠️  No Python VLLM Manager installation found."
    echo
    print_info "💡 If you installed it manually in a custom location, please remove it yourself."
else
    print_warning "📋 The following locations will be removed:"
    for location in "${FOUND_LOCATIONS[@]}"; do
        echo "   • $location"
    done
    echo

    read -p "$(echo -e ${YELLOW}"Continue with removal? (y/N): "${NC})" -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "❌ Uninstallation cancelled."
        exit 0
    fi
fi

print_info "🗑️  Removing Python VLLM Manager..."

# Stop any running services
print_info "🛑 Stopping any running services..."

# Stop user service if exists
if systemctl --user is-active --quiet vllm-manager 2>/dev/null; then
    print_info "   Stopping user service..."
    systemctl --user stop vllm-manager || true
fi

# Stop system service if exists
if systemctl is-active --quiet vllm-manager 2>/dev/null; then
    print_info "   Stopping system service..."
    sudo systemctl stop vllm-manager || true
fi

# Disable services
if systemctl --user is-enabled --quiet vllm-manager 2>/dev/null; then
    print_info "   Disabling user service..."
    systemctl --user disable vllm-manager || true
fi

if systemctl is-enabled --quiet vllm-manager 2>/dev/null; then
    print_info "   Disabling system service..."
    sudo systemctl disable vllm-manager || true
fi

# Remove executable files
print_info "🗑️  Removing executable files..."

for location in "${FOUND_LOCATIONS[@]}"; do
    if [ -f "$location" ] || [ -L "$location" ]; then
        print_info "   Removing: $location"
        rm -f "$location"
    elif [ -d "$location" ]; then
        print_info "   Removing directory: $location"
        rm -rf "$location"
    fi
done

# Remove configuration directories
print_info "🗑️  Removing configuration directories..."

CONFIG_DIRS=(
    "$HOME/.vllm-manager"
    "$HOME/.config/vllm-manager"
)

for config_dir in "${CONFIG_DIRS[@]}"; do
    if [ -d "$config_dir" ]; then
        print_info "   Removing config: $config_dir"
        rm -rf "$config_dir"
    fi
done

# Remove systemd service files
print_info "🗑️  Removing systemd service files..."

for service in "${SYSTEMD_SERVICES[@]}"; do
    if [ -f "$service" ]; then
        if [[ "$service" == /etc/* ]]; then
            print_info "   Removing system service: $service"
            sudo rm -f "$service"
        else
            print_info "   Removing user service: $service"
            rm -f "$service"
        fi
    fi
done

# Reload systemd daemons
print_info "🔄 Reloading systemd daemons..."
systemctl --user daemon-reload 2>/dev/null || true
sudo systemctl daemon-reload 2>/dev/null || true

# Kill any running vLLM processes
print_info "🛑 Killing any running vLLM processes..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -f "vllm_manager.py" 2>/dev/null || true

# Remove from PATH in shell configs
print_info "🧹 Cleaning up shell configurations..."

SHELL_CONFIGS=(
    "$HOME/.bashrc"
    "$HOME/.zshrc"
    "$HOME/.profile"
    "$HOME/.bash_profile"
)

for config in "${SHELL_CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        # Remove vllm-manager PATH entries
        if grep -q "vllm-manager" "$config" 2>/dev/null; then
            print_info "   Cleaning up: $config"
            # Create backup
            cp "$config" "$config.vllm-backup" 2>/dev/null || true

            # Remove vllm-manager lines
            sed -i '/vllm-manager/d' "$config" 2>/dev/null || true
        fi
    fi
done

print_success "✅ Python VLLM Manager uninstalled successfully!"
echo

print_info "📋 Summary of removed items:"
print_info "   • Executable files and symlinks"
print_info "   • Configuration directories"
print_info "   • Systemd service files"
print_info "   • Shell PATH modifications"
print_info "   • Running processes"
echo

print_info "🎉 Ready to install Node.js version!"
echo
print_bold "🚀 Next Steps:"
echo "   1. Run: curl -sSL https://your-domain.com/install-nodejs.sh | bash"
echo "   2. Or: cd nodejs-backend && ./vllm-manager gui"
echo

print_success "🎊 Python version completely removed. Enjoy the Node.js edition!"