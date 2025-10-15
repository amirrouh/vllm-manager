#!/usr/bin/env node

const WebSocket = require('ws');
const readline = require('readline');
const { spawn } = require('child_process');

class VLLMManagerClient {
    constructor() {
        this.ws = null;
        this.models = {};
        this.connected = false;
        this.selectedIndex = 0;
        this.running = true;

        this.setupTerminal();
        this.connect();
    }

    setupTerminal() {
        // Enable raw mode for better keyboard handling
        if (process.stdin.setRawMode) {
            process.stdin.setRawMode(true);
        }
        process.stdin.resume();
        process.stdin.setEncoding('utf8');

        // Handle terminal resize
        process.on('SIGWINCH', () => {
            this.render();
        });

        // Handle exit
        process.on('SIGINT', () => {
            this.cleanup();
            process.exit(0);
        });
    }

    connect() {
        try {
            this.ws = new WebSocket('ws://localhost:3001');

            this.ws.on('open', () => {
                this.connected = true;
                console.clear();
                this.render();
            });

            this.ws.on('message', (data) => {
                const message = JSON.parse(data);
                if (message.type === 'models') {
                    this.models = message.data;
                    this.render();
                }
            });

            this.ws.on('close', () => {
                this.connected = false;
            });

            this.ws.on('error', (error) => {
                console.error('WebSocket error:', error.message);
            });

        } catch (error) {
            console.error('Failed to connect to backend:', error.message);
            console.log('Please make sure the backend server is running: npm start');
            process.exit(1);
        }
    }

    async sendRequest(method, endpoint, data = null) {
        return new Promise((resolve, reject) => {
            const http = require('http');
            const options = {
                hostname: 'localhost',
                port: 3001,
                path: endpoint,
                method: method,
                headers: {
                    'Content-Type': 'application/json'
                }
            };

            const req = http.request(options, (res) => {
                let body = '';
                res.on('data', chunk => body += chunk);
                res.on('end', () => {
                    try {
                        const response = JSON.parse(body);
                        resolve(response);
                    } catch (error) {
                        reject(error);
                    }
                });
            });

            req.on('error', reject);

            if (data) {
                req.write(JSON.stringify(data));
            }

            req.end();
        });
    }

    render() {
        if (!this.connected) {
            console.clear();
            console.log('🔌 Connecting to VLLM Manager Backend...');
            return;
        }

        console.clear();

        const height = process.stdout.rows;
        const width = process.stdout.columns;

        // Header
        const title = '🚀 VLLM MANAGER (Node.js Backend) 🚀';
        console.log(title.padStart((width + title.length) / 2));
        console.log('');

        // Models table
        const models = Object.values(this.models);
        if (models.length === 0) {
            console.log('No models configured. Press A to add a model.');
        } else {
            this.drawModelsTable(models, width);
        }

        // Controls
        console.log('');
        this.drawControls(width);
    }

    drawModelsTable(models, width) {
        const headers = ['Model', 'Status', 'Port', 'Uptime', 'Priority'];
        const colWidths = [20, 12, 8, 10, 8];

        // Header
        let header = '';
        let colX = 2;
        for (let i = 0; i < headers.length; i++) {
            header += headers[i].padEnd(colWidths[i]);
            if (i < headers.length - 1) header += '  ';
        }
        console.log(header);
        console.log('─'.repeat(header.length));

        // Models
        models.forEach((model, index) => {
            const statusIcon = this.getStatusIcon(model.status);
            const uptime = model.uptime_seconds > 0 ? this.formatUptime(model.uptime_seconds) : '-';

            const isSelected = index === this.selectedIndex;
            const prefix = isSelected ? '▶ ' : '  ';
            const colorCode = isSelected ? '\\x1b[7m' : (model.status === 'running' ? '\\x1b[92m' : model.status === 'error' ? '\\x1b[91m' : '\\x1b[90m');
            const resetCode = '\\x1b[0m';

            const row = prefix +
                model.name.padEnd(17) + '  ' +
                statusIcon + ' ' + model.status.padEnd(8) + '  ' +
                model.port.toString().padEnd(6) + '  ' +
                uptime.padEnd(8) + '  ' +
                model.priority.toString();

            console.log(colorCode + row + resetCode);
        });
    }

    drawControls(width) {
        const controls = [
            'CONTROLS:',
            '↑/↓ • Navigate   ENTER • Start/Stop   A • Add   E • Edit   D • Delete',
            'C • Cleanup   Q • Quit'
        ];

        controls.forEach(control => {
            console.log(control.padStart((width + control.length) / 2));
        });
    }

    getStatusIcon(status) {
        const icons = {
            'running': '●',
            'starting': '◐',
            'stopped': '○',
            'error': '✗',
            'unhealthy': '◑'
        };
        return icons[status] || '?';
    }

    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
    }

    async handleInput(key) {
        const models = Object.values(this.models);

        switch (key) {
            case '\\u0003': // Ctrl+C
            case 'q':
            case 'Q':
                this.running = false;
                break;

            case '\\u001b[A': // Up arrow
            case 'k':
            case 'K':
                if (models.length > 0) {
                    this.selectedIndex = Math.max(0, this.selectedIndex - 1);
                    this.render();
                }
                break;

            case '\\u001b[B': // Down arrow
            case 'j':
            case 'J':
                if (models.length > 0) {
                    this.selectedIndex = Math.min(models.length - 1, this.selectedIndex + 1);
                    this.render();
                }
                break;

            case '\\r': // Enter
            case '\\n':
                if (models.length > 0) {
                    const model = models[this.selectedIndex];
                    if (model.status === 'running') {
                        await this.stopModel(model.name);
                    } else {
                        await this.startModel(model.name);
                    }
                }
                break;

            case 'a':
            case 'A':
                await this.addModel();
                break;

            case 'e':
            case 'E':
                if (models.length > 0) {
                    const model = models[this.selectedIndex];
                    await this.editModel(model.name);
                }
                break;

            case 'd':
            case 'D':
                if (models.length > 0) {
                    const model = models[this.selectedIndex];
                    await this.deleteModel(model.name);
                }
                break;

            case 'c':
            case 'C':
                await this.cleanupGPU();
                break;
        }
    }

    async startModel(name) {
        try {
            console.clear();
            console.log(`🚀 Starting model: ${name}`);
            const result = await this.sendRequest('POST', `/api/models/${name}/start`);

            if (result.success) {
                console.log(`✅ ${result.message}`);
            } else {
                console.log(`❌ ${result.error}`);
            }

            setTimeout(() => this.render(), 2000);
        } catch (error) {
            console.log(`❌ Failed to start model: ${error.message}`);
            setTimeout(() => this.render(), 2000);
        }
    }

    async stopModel(name) {
        try {
            console.clear();
            console.log(`🛑 Stopping model: ${name}`);
            const result = await this.sendRequest('POST', `/api/models/${name}/stop`);

            if (result.success) {
                console.log(`✅ ${result.message}`);
            } else {
                console.log(`❌ ${result.error}`);
            }

            setTimeout(() => this.render(), 2000);
        } catch (error) {
            console.log(`❌ Failed to stop model: ${error.message}`);
            setTimeout(() => this.render(), 2000);
        }
    }

    async addModel() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        const question = (prompt) => new Promise(resolve => rl.question(prompt, resolve));

        try {
            console.clear();
            console.log('📝 Add New Model');
            console.log('');

            const name = await question('Model name: ');
            const huggingfaceId = await question('HuggingFace ID: ');
            const port = await question('Port (8001): ');
            const priority = await question('Priority (1-5, default 3): ');

            const modelData = {
                name,
                huggingface_id: huggingfaceId,
                port: parseInt(port) || 8001,
                priority: parseInt(priority) || 3,
                gpu_memory_utilization: 0.3,
                max_model_len: 2048,
                tensor_parallel_size: 1
            };

            const result = await this.sendRequest('POST', '/api/models', modelData);

            if (result.success) {
                console.log(`✅ Model ${name} added successfully`);
            } else {
                console.log(`❌ Failed to add model: ${result.error}`);
            }

        } catch (error) {
            console.log(`❌ Failed to add model: ${error.message}`);
        } finally {
            rl.close();
            setTimeout(() => this.render(), 2000);
        }
    }

    async editModel(name) {
        const model = this.models[name];
        if (!model) return;

        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        const question = (prompt) => new Promise(resolve => rl.question(prompt, resolve));

        try {
            console.clear();
            console.log(`✏️  Edit Model: ${name}`);
            console.log('');

            const newName = await question(`Model name (${model.name}): `);
            const huggingfaceId = await question(`HuggingFace ID (${model.huggingface_id}): `);
            const port = await question(`Port (${model.port}): `);
            const priority = await question(`Priority (1-5, current ${model.priority}): `);

            const updateData = {};
            if (newName && newName !== model.name) updateData.name = newName;
            if (huggingfaceId && huggingfaceId !== model.huggingface_id) updateData.huggingface_id = huggingfaceId;
            if (port && parseInt(port) !== model.port) updateData.port = parseInt(port);
            if (priority && parseInt(priority) !== model.priority) updateData.priority = parseInt(priority);

            if (Object.keys(updateData).length > 0) {
                const result = await this.sendRequest('PUT', `/api/models/${name}`, updateData);

                if (result.success) {
                    console.log(`✅ Model ${name} updated successfully`);
                } else {
                    console.log(`❌ Failed to update model: ${result.error}`);
                }
            } else {
                console.log('No changes made');
            }

        } catch (error) {
            console.log(`❌ Failed to edit model: ${error.message}`);
        } finally {
            rl.close();
            setTimeout(() => this.render(), 2000);
        }
    }

    async deleteModel(name) {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        const question = (prompt) => new Promise(resolve => rl.question(prompt, resolve));

        try {
            console.clear();
            console.log(`🗑️  Delete Model: ${name}`);
            console.log('');

            const confirm = await question(`Are you sure you want to delete '${name}'? (y/N): `);

            if (confirm.toLowerCase() === 'y' || confirm.toLowerCase() === 'yes') {
                const result = await this.sendRequest('DELETE', `/api/models/${name}`);

                if (result.success) {
                    console.log(`✅ Model ${name} deleted successfully`);
                    this.selectedIndex = Math.max(0, this.selectedIndex - 1);
                } else {
                    console.log(`❌ Failed to delete model: ${result.error}`);
                }
            } else {
                console.log('Deletion cancelled');
            }

        } catch (error) {
            console.log(`❌ Failed to delete model: ${error.message}`);
        } finally {
            rl.close();
            setTimeout(() => this.render(), 2000);
        }
    }

    async cleanupGPU() {
        try {
            console.clear();
            console.log('🧹 Cleaning GPU memory...');
            const result = await this.sendRequest('POST', '/api/cleanup');

            if (result.success) {
                console.log(`✅ ${result.message}`);
            } else {
                console.log(`❌ ${result.error}`);
            }

            setTimeout(() => this.render(), 2000);
        } catch (error) {
            console.log(`❌ Failed to cleanup GPU: ${error.message}`);
            setTimeout(() => this.render(), 2000);
        }
    }

    cleanup() {
        if (process.stdin.setRawMode) {
            process.stdin.setRawMode(false);
        }
        if (this.ws) {
            this.ws.close();
        }
        console.log('\\n👋 Goodbye!');
    }

    start() {
        // Handle keyboard input
        process.stdin.on('data', async (key) => {
            if (!this.running) return;

            // Handle arrow keys (escape sequences)
            if (key === '\\u001b') {
                // Buffer for escape sequence
                let buffer = key;

                const timeout = setTimeout(() => {
                    // ESC key pressed
                    this.handleInput(key);
                }, 100);

                process.stdin.once('data', (data) => {
                    clearTimeout(timeout);
                    buffer += data;

                    if (buffer === '\\u001b[A' || buffer === '\\u001b[B') {
                        // Arrow key
                        this.handleInput(buffer);
                    } else {
                        // Other escape sequence
                        this.handleInput(buffer);
                    }
                });
            } else {
                await this.handleInput(key);
            }
        });

        console.log('🔌 Connecting to VLLM Manager Backend...');
    }
}

// Start the client
const client = new VLLMManagerClient();
client.start();