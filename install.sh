#!/bin/bash
# VLLM Manager Installation Script
# Supports both uv and pip for virtual environment setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "🚀 VLLM Manager Installation"
echo "============================="

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ Error: This package is only supported on Linux systems"
    exit 1
fi

# Check for required system packages
echo "🔍 Checking system dependencies..."
MISSING_DEPS=""

# Check for Python development headers
if ! pkg-config --exists python3 2>/dev/null && ! ls /usr/include/python3* 2>/dev/null | grep -q Python.h; then
    MISSING_DEPS="$MISSING_DEPS python3-dev"
fi

# Check for gcc
if ! command -v gcc &> /dev/null; then
    MISSING_DEPS="$MISSING_DEPS gcc"
fi

# Check for g++
if ! command -v g++ &> /dev/null; then
    MISSING_DEPS="$MISSING_DEPS g++"
fi

if [[ -n "$MISSING_DEPS" ]]; then
    echo "⚠️  Missing system dependencies:$MISSING_DEPS"
    echo ""
    echo "Please install them by running:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y python3-dev gcc g++ build-essential"
    echo ""
    echo "For Python 3.12 specifically:"
    echo "  sudo apt-get install -y python3.12-dev"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to install with uv
install_with_uv() {
    echo "📦 Installing with uv..."
    
    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        echo "📥 Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        
        # Verify uv is now available
        if ! command -v uv &> /dev/null; then
            echo "❌ Failed to install uv. Falling back to pip..."
            return 1
        fi
    fi
    
    echo "🔧 Creating virtual environment with uv..."
    cd "$SCRIPT_DIR"
    uv venv .venv
    
    echo "📦 Installing dependencies with uv..."
    uv pip install -r requirements.txt
    
    return 0
}

# Function to install with pip
install_with_pip() {
    echo "📦 Installing with pip..."
    
    # Check if python3 is available
    if ! command -v python3 &> /dev/null; then
        echo "❌ Error: python3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check if venv module is available
    if ! python3 -c "import venv" 2>/dev/null; then
        echo "❌ Error: python3-venv is not installed. Please install it first:"
        echo "   Ubuntu/Debian: sudo apt install python3-venv"
        echo "   RHEL/CentOS:   sudo yum install python3-venv"
        exit 1
    fi
    
    echo "🔧 Creating virtual environment with python3..."
    cd "$SCRIPT_DIR"
    python3 -m venv .venv
    
    echo "📦 Installing dependencies with pip..."
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
}

# Remove existing virtual environment if it exists
if [[ -d "$VENV_DIR" ]]; then
    echo "🧹 Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Try uv first, fallback to pip
if install_with_uv; then
    echo "✅ Installation completed with uv"
elif install_with_pip; then
    echo "✅ Installation completed with pip"
else
    echo "❌ Installation failed"
    exit 1
fi

# Check for .env file
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    if [[ -f "$SCRIPT_DIR/.env.example" ]]; then
        echo "📝 Creating .env file from .env.example..."
        cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
        echo "⚠️  Please edit .env and add your HF_TOKEN for accessing gated models"
    else
        echo "📝 Creating default .env file..."
        cat > "$SCRIPT_DIR/.env" << 'ENVEOF'
# VLLM Manager Environment Variables
# Get your token at: https://huggingface.co/settings/tokens
HF_TOKEN=your_huggingface_token_here
ENVEOF
        echo "⚠️  Please edit .env and add your HF_TOKEN for accessing gated models"
    fi
else
    echo "✅ .env file already exists"
fi

# Create completion script
echo "🔧 Setting up bash completion..."
cat > "$SCRIPT_DIR/.completion" << 'EOF'
# VLLM Manager Bash Completion
# Source this file before using the vllm command: source .completion

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the .env file if it exists
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a  # Export all variables
    source "$SCRIPT_DIR/.env"
    set +a  # Stop exporting
    echo "✅ Environment variables loaded from .env"
fi

_vllm_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    opts="add list start stop remove status cleanup force ui manager"
    
    if [[ ${cur} == -* ]] ; then
        case "${prev}" in
            add)
                COMPREPLY=( $(compgen -W "--port --priority --gpu-memory --max-len --tensor-parallel" -- ${cur}) )
                return 0
                ;;
            *)
                ;;
        esac
    else
        case "${prev}" in
            vllm|./vllm)
                COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
                return 0
                ;;
            start|stop|remove)
                # Get model names from config if it exists
                if [[ -f "models_config.json" ]]; then
                    local models=$(python3 -c "import json; print(' '.join([m['name'] for m in json.load(open('models_config.json', 'r'))]))" 2>/dev/null || echo "")
                    COMPREPLY=( $(compgen -W "${models}" -- ${cur}) )
                fi
                return 0
                ;;
            *)
                ;;
        esac
    fi
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
}

complete -F _vllm_completion vllm
complete -F _vllm_completion ./vllm

echo "💡 VLLM Manager bash completion loaded!"
echo "   Use tab completion with: ./vllm <tab>"
EOF

# Make scripts executable
chmod +x "$SCRIPT_DIR/vllm"
chmod +x "$SCRIPT_DIR/install.sh"

echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "📝 Next Steps:"
echo "1. Source the completion script:"
echo "   source .completion"
echo ""
echo "2. Run the manager:"
echo "   ./vllm                    # Launch terminal UI"
echo "   ./vllm status            # Check system status"
echo "   ./vllm add <name> <id>   # Add a model"
echo ""
echo "💡 Pro tip: Add 'source $(pwd)/.completion' to your ~/.bashrc"
echo "   to automatically load completion in new terminal sessions"
echo ""
echo "📚 For more help: ./vllm --help"