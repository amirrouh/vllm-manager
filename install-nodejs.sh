#!/bin/bash

# VLLM Manager Node.js Installer
# Single command installation: curl -sSL https://your-domain.com/install-nodejs.sh | bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print with colors
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

            NODE.JS EDITION - HIGH PERFORMANCE TERMINAL UI
EOF
    echo -e "${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "❌ Please do not run this installer as root!"
    exit 1
fi

# Installation directory
INSTALL_DIR="$HOME/.vllm-manager-nodejs"
BIN_DIR="$HOME/.local/bin"

print_logo
print_bold "🚀 VLLM Manager Node.js Edition Installer"
echo

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_warning "⚠️  Node.js is not installed."
    echo
    print_info "Installing Node.js..."

    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            # Ubuntu/Debian
            curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
            sudo apt-get install -y nodejs
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL/Fedora
            curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
            sudo yum install -y nodejs
        elif command -v dnf &> /dev/null; then
            # Fedora
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
    NODE_VERSION=$(node --version | cut -d'v' -f2)
    print_success "✅ Node.js $NODE_VERSION found"
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "❌ npm is not installed. Please install npm first."
    exit 1
fi

print_info "📦 Installing VLLM Manager Node.js Edition..."

# Create installation directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download and extract the latest release
RELEASE_URL="https://codeload.github.com/amirrouh/vllm-manager/tar.gz/refs/heads/master"
TEMP_FILE="/tmp/vllm-manager.tar.gz"

print_info "⬇️  Downloading VLLM Manager..."
if command -v curl &> /dev/null; then
    curl -L -o "$TEMP_FILE" "$RELEASE_URL" || {
        print_error "❌ Failed to download VLLM Manager"
        exit 1
    }
elif command -v wget &> /dev/null; then
    wget -O "$TEMP_FILE" "$RELEASE_URL" || {
        print_error "❌ Failed to download VLLM Manager"
        exit 1
    }
else
    print_error "❌ Neither curl nor wget is available"
    exit 1
fi

# Extract
print_info "📂 Extracting files..."
tar -xzf "$TEMP_FILE" --strip-components=1
rm "$TEMP_FILE"

# Copy nodejs-backend files to installation directory
if [ -d "nodejs-backend" ]; then
    cp -r nodejs-backend/* .
    rm -rf nodejs-backend
else
    print_error "❌ nodejs-backend directory not found in release"
    exit 1
fi

# Install dependencies
print_info "📦 Installing Node.js dependencies..."
npm install --production

# Check if Python 3 and vLLM are available
print_info "🔍 Checking Python vLLM environment..."

if ! command -v python3 &> /dev/null; then
    print_warning "⚠️  Python 3 not found. Installing Python 3..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y python3 python3-pip
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3 python3-pip
    fi
fi

# Check for vLLM
VLLM_AVAILABLE=false
if python3 -c "import vllm" &> /dev/null; then
    VLLM_AVAILABLE=true
    print_success "✅ vLLM found"
else
    print_warning "⚠️  vLLM not found. Installing vLLM (this may take 10-20 minutes)..."

    # Create virtual environment
    python3 -m venv "$INSTALL_DIR/.venv"
    source "$INSTALL_DIR/.venv/bin/activate"

    # Install vLLM
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
INSTALL_DIR="$(dirname "$(readlink -f "$0")")"

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

# Add bin directory to PATH if not already there
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "export PATH=\"\$PATH:$BIN_DIR\"" >> "$HOME/.bashrc"
    echo "export PATH=\"\$PATH:$BIN_DIR\"" >> "$HOME/.zshrc" 2>/dev/null || true
    export PATH="$PATH:$BIN_DIR"
fi

# Create configuration directory
CONFIG_DIR="$HOME/.vllm-manager"
mkdir -p "$CONFIG_DIR"

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
echo "   vllm-manager                    # Launch the terminal UI"
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

print_success "🎊 Enjoy your high-performance VLLM Manager!"