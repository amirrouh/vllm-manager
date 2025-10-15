const express = require('express');
const WebSocket = require('ws');
const http = require('http');
const cors = require('cors');
const axios = require('axios');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

class VLLMManagerBackend {
    constructor() {
        this.app = express();
        this.server = http.createServer(this.app);
        this.wss = new WebSocket.Server({ server: this.server });

        this.app.use(cors());
        this.app.use(express.json());

        this.clients = new Set();
        this.configDir = path.join(os.homedir(), '.vllm-manager');
        this.configFile = path.join(this.configDir, 'models.json');

        this.models = {};
        this.monitoringInterval = null;

        this.setupRoutes();
        this.setupWebSocket();
        this.loadConfig();
        this.startMonitoring();
    }

    setupRoutes() {
        // Get all models
        this.app.get('/api/models', (req, res) => {
            res.json(this.models);
        });

        // Add model
        this.app.post('/api/models', (req, res) => {
            try {
                const { name, huggingface_id, port, priority = 3, gpu_memory_utilization = 0.3, max_model_len = 2048, tensor_parallel_size = 1 } = req.body;

                if (this.models[name]) {
                    return res.status(400).json({ error: 'Model already exists' });
                }

                const model = {
                    name,
                    huggingface_id,
                    port: parseInt(port),
                    priority: parseInt(priority),
                    gpu_memory_utilization: parseFloat(gpu_memory_utilization),
                    max_model_len: parseInt(max_model_len),
                    tensor_parallel_size: parseInt(tensor_parallel_size),
                    status: 'stopped',
                    pid: null,
                    gpu_memory_mb: 0,
                    cpu_percent: 0,
                    uptime_seconds: 0,
                    start_time: null,
                    last_error: null
                };

                this.models[name] = model;
                this.saveConfig();
                this.broadcastUpdate();

                res.json({ success: true, model });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Update model
        this.app.put('/api/models/:name', (req, res) => {
            try {
                const { name } = req.params;
                const updates = req.body;

                if (!this.models[name]) {
                    return res.status(404).json({ error: 'Model not found' });
                }

                // Check if name is being changed and new name already exists
                if (updates.name && updates.name !== name && this.models[updates.name]) {
                    return res.status(400).json({ error: 'Model with new name already exists' });
                }

                const oldName = name;
                const newName = updates.name || name;

                // Handle model rename
                if (oldName !== newName) {
                    this.models[newName] = { ...this.models[oldName], ...updates };
                    delete this.models[oldName];
                } else {
                    this.models[name] = { ...this.models[name], ...updates };
                }

                this.saveConfig();
                this.broadcastUpdate();

                res.json({ success: true, model: this.models[newName] });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Delete model
        this.app.delete('/api/models/:name', (req, res) => {
            try {
                const { name } = req.params;

                if (!this.models[name]) {
                    return res.status(404).json({ error: 'Model not found' });
                }

                // Stop model if running
                if (this.models[name].status === 'running' && this.models[name].pid) {
                    this.stopModel(name);
                }

                delete this.models[name];
                this.saveConfig();
                this.broadcastUpdate();

                res.json({ success: true });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Start model
        this.app.post('/api/models/:name/start', async (req, res) => {
            try {
                const { name } = req.params;
                const result = await this.startModel(name);
                res.json(result);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Stop model
        this.app.post('/api/models/:name/stop', (req, res) => {
            try {
                const { name } = req.params;
                const result = this.stopModel(name);
                res.json(result);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Get GPU info
        this.app.get('/api/gpu', async (req, res) => {
            try {
                const gpuInfo = await this.getGPUInfo();
                res.json(gpuInfo);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Cleanup GPU
        this.app.post('/api/cleanup', (req, res) => {
            try {
                const result = this.cleanupGPU();
                res.json(result);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Serve static files for the terminal UI
        this.app.use(express.static(path.join(__dirname, '../public')));
    }

    setupWebSocket() {
        this.wss.on('connection', (ws) => {
            this.clients.add(ws);
            console.log('Client connected');

            // Send initial data
            ws.send(JSON.stringify({ type: 'models', data: this.models }));

            ws.on('close', () => {
                this.clients.delete(ws);
                console.log('Client disconnected');
            });
        });
    }

    broadcastUpdate() {
        const message = JSON.stringify({ type: 'models', data: this.models });
        this.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(message);
            }
        });
    }

    loadConfig() {
        try {
            if (fs.existsSync(this.configFile)) {
                const data = JSON.parse(fs.readFileSync(this.configFile, 'utf8'));
                this.models = data.models || {};
                console.log(`Loaded ${Object.keys(this.models).length} models from config`);
            }
        } catch (error) {
            console.error('Error loading config:', error);
        }
    }

    saveConfig() {
        try {
            if (!fs.existsSync(this.configDir)) {
                fs.mkdirSync(this.configDir, { recursive: true });
            }
            const data = { models: this.models };
            fs.writeFileSync(this.configFile, JSON.stringify(data, null, 2));
            console.log('Configuration saved');
        } catch (error) {
            console.error('Error saving config:', error);
        }
    }

    async startModel(name) {
        const model = this.models[name];
        if (!model) {
            return { success: false, error: 'Model not found' };
        }

        if (model.status === 'running') {
            return { success: false, error: 'Model already running' };
        }

        try {
            model.status = 'starting';
            this.broadcastUpdate();

            // Check GPU availability
            const gpuInfo = await this.getGPUInfo();
            if (!gpuInfo.length) {
                model.status = 'error';
                model.last_error = 'No GPU available';
                this.broadcastUpdate();
                return { success: false, error: 'No GPU available' };
            }

            // Kill any process using the target port
            await this.killProcessOnPort(model.port);

            // Start the vLLM process
            const vllmCmd = [
                'python3', '-m', 'vllm.entrypoints.openai.api_server',
                '--model', model.huggingface_id,
                '--host', '0.0.0.0',
                '--port', model.port.toString(),
                '--trust-remote-code',
                '--gpu-memory-utilization', model.gpu_memory_utilization.toString(),
                '--max-model-len', model.max_model_len.toString(),
                '--tensor-parallel-size', model.tensor_parallel_size.toString(),
                '--enforce-eager',
                '--disable-log-requests'
            ];

            const process = spawn('python3', vllmCmd.slice(1), {
                env: { ...process.env, CUDA_VISIBLE_DEVICES: '0' },
                stdio: 'pipe'
            });

            model.pid = process.pid;
            model.start_time = new Date().toISOString();
            this.broadcastUpdate();

            // Wait for health check
            for (let i = 0; i < 30; i++) {
                await new Promise(resolve => setTimeout(resolve, 2000));

                if (await this.checkModelHealth(model.port)) {
                    model.status = 'running';
                    this.broadcastUpdate();
                    return { success: true, message: `Model ${name} started successfully on port ${model.port}` };
                }

                if (process.exitCode !== null) {
                    model.status = 'error';
                    model.last_error = `Process exited with code ${process.exitCode}`;
                    model.pid = null;
                    this.broadcastUpdate();
                    return { success: false, error: model.last_error };
                }
            }

            // Timeout
            process.kill();
            model.status = 'error';
            model.last_error = 'Timeout during startup';
            model.pid = null;
            this.broadcastUpdate();
            return { success: false, error: 'Model failed to start within timeout' };

        } catch (error) {
            model.status = 'error';
            model.last_error = error.message;
            model.pid = null;
            this.broadcastUpdate();
            return { success: false, error: error.message };
        }
    }

    stopModel(name) {
        const model = this.models[name];
        if (!model) {
            return { success: false, error: 'Model not found' };
        }

        if (model.status !== 'running') {
            return { success: false, error: 'Model not running' };
        }

        try {
            if (model.pid) {
                process.kill(model.pid, 'SIGTERM');
                setTimeout(() => {
                    try {
                        process.kill(model.pid, 'SIGKILL');
                    } catch (e) {
                        // Process already dead
                    }
                }, 1000);
            }

            model.status = 'stopped';
            model.pid = null;
            model.start_time = null;
            this.broadcastUpdate();

            return { success: true, message: `Model ${name} stopped` };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async checkModelHealth(port) {
        try {
            const response = await axios.get(`http://localhost:${port}/v1/models`, { timeout: 5000 });
            return response.status === 200;
        } catch (error) {
            return false;
        }
    }

    async getGPUInfo() {
        try {
            const { stdout } = await this.execCommand('nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits');

            const lines = stdout.trim().split('\\n');
            return lines.map(line => {
                const [index, name, memoryUsed, memoryTotal, utilization, temperature] = line.split(',').map(s => s.trim());
                return {
                    index: parseInt(index),
                    name,
                    memoryUsed: parseFloat(memoryUsed),
                    memoryTotal: parseFloat(memoryTotal),
                    utilization: parseFloat(utilization),
                    temperature: temperature !== '[Not Supported]' ? parseInt(temperature) : null
                };
            });
        } catch (error) {
            console.error('Error getting GPU info:', error);
            return [];
        }
    }

    async killProcessOnPort(port) {
        try {
            const { stdout } = await this.execCommand(`lsof -ti :${port}`);
            if (stdout.trim()) {
                const pid = parseInt(stdout.trim());
                process.kill(pid, 'SIGTERM');
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        } catch (error) {
            // Port not in use or other error
        }
    }

    cleanupGPU() {
        try {
            // Kill all vLLM processes
            spawn('pkill', ['-f', 'vllm.entrypoints.openai.api_server']);

            setTimeout(() => {
                spawn('pkill', ['-9', '-f', 'vllm.entrypoints.openai.api_server']);
            }, 2000);

            return { success: true, message: 'GPU cleanup initiated' };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    execCommand(command) {
        return new Promise((resolve, reject) => {
            require('child_process').exec(command, (error, stdout, stderr) => {
                if (error) {
                    reject(error);
                } else {
                    resolve({ stdout, stderr });
                }
            });
        });
    }

    startMonitoring() {
        this.monitoringInterval = setInterval(async () => {
            try {
                // Update model stats
                for (const [name, model] of Object.entries(this.models)) {
                    if (model.status === 'running' && model.pid) {
                        try {
                            // Check if process is still running
                            process.kill(model.pid, 0); // Signal 0 just checks if process exists

                            // Update uptime
                            if (model.start_time) {
                                const startTime = new Date(model.start_time);
                                model.uptime_seconds = Math.floor((new Date() - startTime) / 1000);
                            }

                            // Check health
                            if (!(await this.checkModelHealth(model.port))) {
                                model.status = 'unhealthy';
                                model.last_error = 'Health check failed';
                            }
                        } catch (error) {
                            model.status = 'error';
                            model.last_error = 'Process died unexpectedly';
                            model.pid = null;
                        }
                    }
                }

                this.broadcastUpdate();
            } catch (error) {
                console.error('Error in monitoring:', error);
            }
        }, 5000); // Check every 5 seconds
    }

    start(port = 3001) {
        this.server.listen(port, () => {
            console.log(`VLLM Manager Backend running on port ${port}`);
            console.log(`WebSocket server ready for connections`);
        });
    }

    stop() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
        }
        this.server.close();
    }
}

// Start the server
const backend = new VLLMManagerBackend();
backend.start();

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\\nShutting down gracefully...');
    backend.stop();
    process.exit(0);
});

module.exports = VLLMManagerBackend;