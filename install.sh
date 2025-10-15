#!/bin/bash

# VLLM Manager Smart Installer - Automatically handles caching and updates
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}VLLM Manager Smart Installer${NC}"
echo -e "${BLUE}Automatically handles caching and updates${NC}"

# Get latest version info with cache busting
TIMESTAMP=$(date +%s)
echo -e "${YELLOW}📡 Checking for latest version...${NC}"

# Download the latest installer directly
LATEST_URL="https://raw.githubusercontent.com/amirrouh/vllm-manager/master/install-latest.sh"
echo -e "${YELLOW}⬇️  Downloading latest installer...${NC}"

# Create temp file
TEMP_INSTALLER="/tmp/vllm-installer-$TIMESTAMP.sh"

# Download with cache busting
if curl -fsSL "${LATEST_URL}?t=${TIMESTAMP}" -o "$TEMP_INSTALLER"; then
    echo -e "${GREEN}✅ Download successful${NC}"
    chmod +x "$TEMP_INSTALLER"
    
    # Execute the downloaded installer
    echo -e "${YELLOW}🚀 Launching installer...${NC}"
    exec bash "$TEMP_INSTALLER" "$@"
else
    echo -e "${RED}❌ Failed to download installer${NC}"
    echo -e "${YELLOW}Trying fallback method...${NC}"
    
    # Fallback: run embedded installer
    echo -e "${YELLOW}🔄 Using embedded installer...${NC}"
    
    # Embedded installer content (simplified version)
    INSTALL_DIR="$HOME/.vllm-manager"
    BIN_DIR="$HOME/.local/bin"
    
    echo -e "${BLUE}🔧 Creating directories...${NC}"
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"
    
    echo -e "${BLUE}📥 Downloading from GitHub...${NC}"
    cd /tmp
    rm -rf vllm-manager-temp
    git clone https://github.com/amirrouh/vllm-manager.git vllm-manager-temp
    cd vllm-manager-temp
    
    echo -e "${BLUE}📦 Installing files...${NC}"
    cp -r * "$INSTALL_DIR/"
    cd "$INSTALL_DIR"
    
    echo -e "${BLUE}⚙️  Setting up environment...${NC}"
    cat > pyproject.toml << 'INNEREOF'
[project]
name = "vllm-manager"
version = "0.1.0"
description = "Zero-Setup vLLM Model Management"
requires-python = ">=3.8"
dependencies = ["vllm"]
INNEREOF
    
    if command -v uv >/dev/null 2>&1; then
        uv add vllm
    else
        echo -e "${YELLOW}Installing uv...${NC}"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source ~/.bashrc
        uv add vllm
    fi
    
    echo -e "${BLUE}🔧 Creating vm command...${NC}"
    cat > "$BIN_DIR/vm" << 'INNEREOF'
#!/bin/bash
cd "$HOME/.vllm-manager"
if [[ "$1" == "help" ]] || [[ -z "$1" ]]; then
    echo "🚀 VLLM Manager - Use 'vm gui' to launch"
    exit 0
fi
exec uv run python vllm_manager.py "$@"
INNEREOF
    
    chmod +x "$BIN_DIR/vm"
    
    echo -e "${GREEN}🎉 Installation Complete!${NC}"
    echo -e "${YELLOW}Use: vm gui${NC}"
    
    # Clean up
    rm -rf /tmp/vllm-manager-temp
    rm -f "$TEMP_INSTALLER"
fi
