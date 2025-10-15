#!/bin/bash

# VLLM Manager Node.js Edition - Complete Installer
# This script installs the Node.js version and removes any Python version

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
    echo -e "${BLUE}"
    cat << 'EOF'
    ____              __    ____            _
   / __ \____  __  __/ /   / __ \____  ____(_)___  ____  _____
  / /_/ / __ \/ / / / /   / /_/ / __ \/ __/ / __ \/ __ \/ ___/
 / ____/ /_/ / /_/ / /___/ ____/ /_/ / /_/ / / / / /_/ (__  )
/_/    \____/\__,_/_____/_/    \____/\__/_/_/ /_/\____/____/

            NODE.JS EDITION - COMPLETE INSTALLER
EOF
    echo -e "${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "❌ Please do not run this installer as root!"
    exit 1
fi

print_logo
print_bold "🚀 VLLM Manager Node.js Edition - Complete Installer"
echo

# Check for existing Python installation
PYTHON_INSTALLED=false
PYTHON_LOCATIONS=()

# Check common Python installation locations
PYTHON_LOCATIONS=(
    "$HOME/.local/bin/vm"
    "$HOME/.local/bin/vllm-manager"
    "$HOME/apps/vllm-manager"
    "$HOME/vllm-manager"
    "$HOME/.vllm-manager"
)

for location in "${PYTHON_LOCATIONS[@]}"; do
    if [ -e "$location" ]; then
        PYTHON_INSTALLED=true
        break
    fi
done

if [ "$PYTHON_INSTALLED" = true ]; then
    print_warning "⚠️  Python VLLM Manager installation detected!"
    echo
    print_info "📋 Found Python installation components:"

    for location in "${PYTHON_LOCATIONS[@]}"; do
        if [ -e "$location" ]; then
            echo "   • $location"
        fi
    done

    echo
    print_warning "This installer will:"
    echo "   1. Remove the Python version"
    echo "   2. Install the Node.js version"
    echo "   3. Migrate your configurations"
    echo

    read -p "$(echo -e ${YELLOW}"Continue with migration to Node.js? (y/N): "${NC})" -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "❌ Installation cancelled."
        exit 0
    fi

    # Run uninstallation
    print_info "🗑️  Removing Python version..."

    # Kill any running Python VLLM processes
    pkill -f "vllm_manager.py" 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

    # Remove Python installation
    for location in "${PYTHON_LOCATIONS[@]}"; do
        if [ -e "$location" ]; then
            print_info "   Removing: $location"
            rm -rf "$location"
        fi
    done

    # Remove Python config if it exists
    if [ -d "$HOME/.vllm-manager" ]; then
        # Backup configurations before removing
        if [ -f "$HOME/.vllm-manager/models.json" ]; then
            print_info "   Backing up configurations..."
            mkdir -p "$HOME/.vllm-manager-backup"
            cp -r "$HOME/.vllm-manager" "$HOME/.vllm-manager-backup/"
        fi
    fi

    print_success "✅ Python version removed successfully"
    echo
else
    print_success "✅ No Python version detected - proceeding with fresh installation"
    echo
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_warning "⚠️  Node.js is not installed."
    echo
    print_info "Installing Node.js..."

    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            # Ubuntu/Debian
            print_info "   Installing Node.js on Ubuntu/Debian..."
            curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
            sudo apt-get install -y nodejs
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL/Fedora
            print_info "   Installing Node.js on CentOS/RHEL..."
            curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
            sudo yum install -y nodejs
        elif command -v dnf &> /dev/null; then
            # Fedora
            print_info "   Installing Node.js on Fedora..."
            curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
            sudo dnf install -y nodejs
        else
            print_error "❌ Unable to install Node.js automatically."
            print_info "Please install Node.js manually: https://nodejs.org/"
            exit 1
        fi
    else
        print_error "❌ Unsupported operating system for automatic Node.js installation."
        print_info "Please install Node.js manually: https://nodejs.org/"
        exit 1
    fi
else
    NODE_VERSION=$(node --version)
    print_success "✅ Node.js $NODE_VERSION found"
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Installation directory
INSTALL_DIR="$HOME/.vllm-manager-nodejs"
BIN_DIR="$HOME/.local/bin"

print_info "📦 Installing VLLM Manager Node.js Edition..."

# Remove any existing Node.js installation
if [ -d "$INSTALL_DIR" ]; then
    print_info "   Removing existing Node.js installation..."
    rm -rf "$INSTALL_DIR"
fi

# Create installation directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Get current directory for copying files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy files from current directory
print_info "   Copying application files..."
if [ -d "$SCRIPT_DIR/nodejs-backend" ]; then
    cp -r "$SCRIPT_DIR/nodejs-backend"/* "$INSTALL_DIR/"
else
    print_error "❌ Node.js backend directory not found"
    print_info "   Please run this script from the vllm-manager directory"
    exit 1
fi

# Install dependencies
print_info "   Installing Node.js dependencies..."
npm install --production

# Check for Python 3 and vLLM
print_info "🔍 Checking Python vLLM environment..."

VLLM_AVAILABLE=false
VENV_PATH="$INSTALL_DIR/.venv"

if python3 -c "import vllm" &> /dev/null; then
    VLLM_AVAILABLE=true
    print_success "✅ vLLM found in system Python"
else
    print_warning "⚠️  vLLM not found. Installing vLLM (this may take 10-20 minutes)..."

    # Create virtual environment
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"

    # Install vLLM
    pip install --upgrade pip
    pip install vllm || {
        print_error "❌ Failed to install vLLM automatically."
        print_info "Please install vLLM manually: pip install vllm"
        exit 1
    }

    VLLM_AVAILABLE=true
    print_success "✅ vLLM installed successfully"
fi

# Create bin directory and symlink
mkdir -p "$BIN_DIR"

# Create the main executable
cat > "$INSTALL_DIR/vllm-manager" << 'EOF'
#!/bin/bash
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment exists
if [ -d "$INSTALL_DIR/.venv" ]; then
    source "$INSTALL_DIR/.venv/bin/activate"
fi

# Check if Node.js modules are installed
if [ ! -d "$INSTALL_DIR/node_modules" ]; then
    echo "❌ Node.js modules not found. Please run the installer again."
    exit 1
fi

# Start the Node.js application
cd "$INSTALL_DIR"
node src/app.js "$@"
EOF

chmod +x "$INSTALL_DIR/vllm-manager"

# Create symlink in user bin directory
if [ -L "$BIN_DIR/vllm-manager" ]; then
    unlink "$BIN_DIR/vllm-manager"
fi
ln -s "$INSTALL_DIR/vllm-manager" "$BIN_DIR/vllm-manager"

# Also create 'vm' symlink for compatibility
if [ -L "$BIN_DIR/vm" ]; then
    unlink "$BIN_DIR/vm"
fi
ln -s "$INSTALL_DIR/vllm-manager" "$BIN_DIR/vm"

# Add bin directory to PATH if not already there
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "export PATH=\"\$PATH:$BIN_DIR\"" >> "$HOME/.bashrc"
    echo "export PATH=\"\$PATH:$BIN_DIR\"" >> "$HOME/.zshrc" 2>/dev/null || true
    export PATH="$PATH:$BIN_DIR"
fi

# Create configuration directory and migrate settings
CONFIG_DIR="$HOME/.vllm-manager"
mkdir -p "$CONFIG_DIR"

# Migrate configurations from Python version if backup exists
if [ -d "$HOME/.vllm-manager-backup" ]; then
    print_info "🔄 Migrating configurations from Python version..."

    if [ -f "$HOME/.vllm-manager-backup/models.json" ]; then
        cp "$HOME/.vllm-manager-backup/models.json" "$CONFIG_DIR/"
        print_success "✅ Model configurations migrated"
    fi

    if [ -f "$HOME/.vllm-manager-backup/.env" ]; then
        cp "$HOME/.vllm-manager-backup/.env" "$CONFIG_DIR/"
        print_success "✅ Environment settings migrated"
    fi

    # Clean up backup
    rm -rf "$HOME/.vllm-manager-backup"
fi

# Create initial config if it doesn't exist
if [ ! -f "$CONFIG_DIR/models.json" ]; then
    cat > "$CONFIG_DIR/models.json" << 'EOF'
{
  "models": [
    {
      "name": "bert-ner",
      "huggingface_id": "dslim/bert-base-NER",
      "port": 8001,
      "priority": 3,
      "gpu_memory_utilization": 0.3,
      "max_model_len": 512,
      "tensor_parallel_size": 1
    }
  ]
}
EOF
    print_success "✅ Default configuration created"
fi

# Create systemd service for auto-start (optional)
SERVICE_FILE="$HOME/.config/systemd/user/vllm-manager.service"
mkdir -p "$(dirname "$SERVICE_FILE")"

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=VLLM Manager Node.js Backend
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/node src/server.js
Restart=always
RestartSec=10
Environment=NODE_ENV=production

[Install]
WantedBy=default.target
EOF

# Reload systemd daemon
systemctl --user daemon-reload 2>/dev/null || true

print_success "✅ Installation completed successfully!"
echo

print_bold "🎉 VLLM Manager Node.js Edition is now installed!"
echo

print_info "📋 Usage:"
echo "   vllm-manager                    # Launch the modern terminal UI"
echo "   vm                               # Shortcut alias"
echo "   vllm-manager start <model>      # Start a model"
echo "   vllm-manager stop <model>       # Stop a model"
echo "   vllm-manager list               # List all models"
echo "   vllm-manager add <name> <hf_id> # Add a model"
echo "   vllm-manager remove <model>     # Remove a model"
echo "   vllm-manager status             # Show system status"
echo "   vllm-manager cleanup            # Clean GPU memory"
echo "   vllm-manager --help             # Show help"
echo

print_info "🔧 Configuration files:"
echo "   Models config: $CONFIG_DIR/models.json"
echo "   Log files:   $CONFIG_DIR/logs/"
echo "   Install dir:  $INSTALL_DIR"
echo

print_info "🚀 To start using VLLM Manager:"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    print_warning "⚠️  You may need to restart your terminal or run:"
    echo "   export PATH=\"\$PATH:$BIN_DIR\""
    echo
fi
echo "   vllm-manager"
echo

print_info "💡 For auto-start backend on boot:"
echo "   systemctl --user enable vllm-manager.service"
echo "   systemctl --user start vllm-manager.service"
echo

print_success "🎊 Migration complete! Enjoy your high-performance VLLM Manager!"