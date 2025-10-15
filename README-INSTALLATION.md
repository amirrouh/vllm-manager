# VLLM Manager - Installation Guide

## 🚀 Quick Installation Options

### Option 1: Local Installation (Recommended)

Since you already have the files locally, just run:

```bash
# Complete migration from Python to Node.js
./install-nodejs-complete.sh
```

This will:
- ✅ Remove Python version safely
- ✅ Install Node.js version
- ✅ Migrate your configurations
- ✅ Set up everything automatically

### Option 2: Remote Installation

For remote installation, you need to host the installer somewhere. Here are your options:

#### A. GitHub Pages (Free)

1. **Create a GitHub repository** with the installer files
2. **Enable GitHub Pages**
3. **Use the raw file URL**

```bash
curl -sSL https://raw.githubusercontent.com/amirrouh/vllm-manager/master/install-nodejs-complete.sh | bash
```

#### B. Self-Hosted Web Server

If you have your own domain/server:

1. **Upload the installer** to your web server
2. **Make it publicly accessible**
3. **Use your domain URL**

```bash
curl -sSL https://your-domain.com/install-nodejs-complete.sh | bash
```

#### C. Use a File Hosting Service

- **GitHub Gist**: Create a gist with the installer
- **Paste.ee**: Upload and get shareable link
- **File.io**: Temporary file hosting

## 📁 Required Files for Remote Installation

To host the installer remotely, you need these files:

```
vllm-manager/
├── install-nodejs-complete.sh          # Main installer
├── nodejs-backend/                      # Node.js application
│   ├── package.json
│   ├── src/
│   │   ├── app.js
│   │   ├── ui.js
│   │   ├── manager.js
│   │   └── server.js
│   └── vllm-manager
└── README-INSTALLATION.md
```

## 🛠️ Setting Up Remote Installation

### Method 1: GitHub Repository (Easiest)

1. **Create a new GitHub repository**: `vllm-manager-nodejs`
2. **Upload all files** from the `nodejs-backend` directory
3. **Upload the installers** (`install-nodejs-complete.sh`)
4. **Enable GitHub Pages** in repository settings
5. **Use the raw URL**:

```bash
# Replace with your actual GitHub username and repo
curl -sSL https://raw.githubusercontent.com/yourusername/vllm-manager-nodejs/main/install-nodejs-complete.sh | bash
```

### Method 2: Personal Web Server

If you have a domain and web server:

1. **Upload files to your web server**
2. **Make sure the installer is executable and accessible**
3. **Update the URL** in your documentation

```bash
# Replace with your actual domain
curl -sSL https://your-domain.com/vllm-manager/install-nodejs-complete.sh | bash
```

### Method 3: GitHub Gist (Quick & Easy)

1. **Copy the installer content**
2. **Create a new GitHub Gist**: https://gist.github.com
3. **Paste the installer code**
4. **Get the raw URL** and use it

## 🔄 Currently Available Installation

Since you have the files locally, you can just run:

```bash
# From your vllm-manager directory
./install-nodejs-complete.sh
```

This will:
- Detect your Python installation
- Safely remove it with configuration backup
- Install the Node.js version
- Migrate all your settings
- Set up proper symlinks and PATH

## 📦 What the Installer Does

### ✅ Python Removal
- Stops running services
- Removes executables (`vm`, `vllm-manager`)
- Backs up configurations to `~/.vllm-manager-backup/`
- Cleans shell configs (.bashrc, .zshrc)
- Removes systemd services

### ✅ Node.js Installation
- Checks for Node.js (installs if needed)
- Installs npm dependencies
- Sets up Python virtual environment with vLLM
- Creates symlinks in `~/.local/bin/`
- Updates PATH in shell configs
- Creates systemd service
- Migrates configurations

### ✅ Configuration Migration
- Preserves `models.json`
- Preserves HuggingFace tokens
- Maintains all your model settings

## 🎯 Recommendation

For now, **use the local installation**:

```bash
cd /home/ubuntu/apps/vllm-manager
./install-nodejs-complete.sh
```

This is the safest and most reliable method since you already have all the files locally.

Later, if you want to share with others or install on multiple machines, you can set up a remote repository.

---

**🚀 Ready to migrate? Just run `./install-nodejs-complete.sh`!**