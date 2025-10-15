const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');
const axios = require('axios');
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

class VLLMManager {
    constructor() {
        this.configDir = path.join(os.homedir(), '.vllm-manager');
        this.configFile = path.join(this.configDir, 'models.json');
        this.envFile = path.join(this.configDir, '.env');
        this.models = new Map();
        this.monitoringInterval = null;

        this.init();
    }

    async init() {
        // Ensure config directory exists
        await fs.mkdir(this.configDir, { recursive: true });

        // Load models configuration
        await this.loadModels();

        // Start monitoring
        this.startMonitoring();
    }

    async loadModels() {
        try {
            if (await this.fileExists(this.configFile)) {
                const data = await fs.readFile(this.configFile, 'utf8');
                const config = JSON.parse(data);

                if (config.models && Array.isArray(config.models)) {
                    this.models.clear();
                    config.models.forEach(model => {
                        // Add runtime properties
                        model.status = model.status || 'stopped';
                        model.pid = model.pid || null;
                        model.gpu_memory_mb = model.gpu_memory_mb || 0;
                        model.cpu_percent = model.cpu_percent || 0;
                        model.uptime_seconds = model.uptime_seconds || 0;
                        model.start_time = model.start_time || null;
                        model.last_error = model.last_error || null;

                        this.models.set(model.name, model);
                    });
                }
            }
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    async saveModels() {
        try {
            const config = {
                models: Array.from(this.models.values()).map(model => {
                    // Remove runtime properties for saving
                    const { status, pid, gpu_memory_mb, cpu_percent, uptime_seconds, start_time, last_error, ...config } = model;
                    return config;
                })
            };

            await fs.writeFile(this.configFile, JSON.stringify(config, null, 2));
        } catch (error) {
            console.error('Error saving models:', error);
            throw error;
        }
    }

    async fileExists(filePath) {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }

    async getAllModels() {
        return Array.from(this.models.values());
    }

    async getModel(name) {
        return this.models.get(name);
    }

    async addModel(modelConfig) {
        try {
            // Validate model doesn't already exist
            if (this.models.has(modelConfig.name)) {
                return { success: false, error: 'Model already exists' };
            }

            // Create model object with runtime properties
            const model = {
                ...modelConfig,
                status: 'stopped',
                pid: null,
                gpu_memory_mb: 0,
                cpu_percent: 0,
                uptime_seconds: 0,
                start_time: null,
                last_error: null
            };

            this.models.set(modelConfig.name, model);
            await this.saveModels();

            return { success: true, model };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async updateModel(name, updates) {
        try {
            const model = this.models.get(name);
            if (!model) {
                return { success: false, error: 'Model not found' };
            }

            // Handle model rename
            const newName = updates.name || name;
            if (newName !== name && this.models.has(newName)) {
                return { success: false, error: 'Model with new name already exists' };
            }

            // Check if model needs to be restarted
            const needsRestart = model.status === 'running' && (
                updates.huggingface_id && updates.huggingface_id !== model.huggingface_id ||
                updates.port && updates.port !== model.port
            );

            // Stop model if it needs restart
            if (needsRestart) {
                await this.stopModel(name);
            }

            // Update model properties
            const updatedModel = { ...model, ...updates };

            // Handle rename
            if (newName !== name) {
                this.models.delete(name);
                this.models.set(newName, updatedModel);
            } else {
                this.models.set(name, updatedModel);
            }

            await this.saveModels();

            // Restart if needed
            if (needsRestart) {
                await this.startModel(newName);
            }

            return { success: true, model: updatedModel };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async removeModel(name) {
        try {
            const model = this.models.get(name);
            if (!model) {
                return { success: false, error: 'Model not found' };
            }

            // Stop model if running
            if (model.status === 'running') {
                await this.stopModel(name);
            }

            this.models.delete(name);
            await this.saveModels();

            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async startModel(name) {
        try {
            const model = this.models.get(name);
            if (!model) {
                return { success: false, error: 'Model not found' };
            }

            if (model.status === 'running') {
                return { success: false, error: 'Model already running' };
            }

            // Check GPU availability
            const gpuInfo = await this.getGPUInfo();
            if (gpuInfo.length === 0) {
                return { success: false, error: 'No GPU available' };
            }

            // Kill any process using the target port
            await this.killProcessOnPort(model.port);

            // Set model status to starting
            model.status = 'starting';
            model.start_time = new Date().toISOString();
            model.last_error = null;

            // Build vLLM command
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

            // Prepare environment
            const env = { ...process.env };

            // Add HF_TOKEN if available
            const hfToken = await this.getHFToken();
            if (hfToken) {
                env.HF_TOKEN = hfToken;
            }

            // Add CUDA_VISIBLE_DEVICES
            env.CUDA_VISIBLE_DEVICES = '0';

            // Start the vLLM process
            const vllmProcess = spawn('python3', vllmCmd.slice(1), {
                env,
                stdio: 'pipe',
                detached: false
            });

            model.pid = vllmProcess.pid;

            // Handle process output
            vllmProcess.stdout.on('data', (data) => {
                // Log output if needed for debugging
                console.log(`vLLM [${name}]:`, data.toString());
            });

            vllmProcess.stderr.on('data', (data) => {
                console.error(`vLLM Error [${name}]:`, data.toString());
            });

            vllmProcess.on('exit', (code, signal) => {
                if (model.status === 'running') {
                    model.status = 'error';
                    model.last_error = `Process exited with code ${code}`;
                    model.pid = null;
                }
            });

            // Wait for health check
            const maxAttempts = 30;
            for (let i = 0; i < maxAttempts; i++) {
                await new Promise(resolve => setTimeout(resolve, 2000));

                if (vllmProcess.exitCode !== null) {
                    model.status = 'error';
                    model.last_error = `Process exited with code ${vllmProcess.exitCode}`;
                    model.pid = null;
                    return { success: false, error: model.last_error };
                }

                if (await this.checkModelHealth(model.port)) {
                    model.status = 'running';
                    return { success: true, message: `Model ${name} started successfully on port ${model.port}` };
                }
            }

            // Timeout
            vllmProcess.kill();
            model.status = 'error';
            model.last_error = 'Timeout during startup';
            model.pid = null;
            return { success: false, error: 'Model failed to start within timeout' };

        } catch (error) {
            const model = this.models.get(name);
            if (model) {
                model.status = 'error';
                model.last_error = error.message;
                model.pid = null;
            }
            return { success: false, error: error.message };
        }
    }

    async stopModel(name) {
        try {
            const model = this.models.get(name);
            if (!model) {
                return { success: false, error: 'Model not found' };
            }

            if (model.status !== 'running') {
                return { success: false, error: 'Model not running' };
            }

            if (model.pid) {
                try {
                    // Send SIGTERM
                    process.kill(model.pid, 'SIGTERM');

                    // Wait a bit, then send SIGKILL if still running
                    setTimeout(() => {
                        try {
                            process.kill(model.pid, 'SIGKILL');
                        } catch (e) {
                            // Process already dead
                        }
                    }, 1000);
                } catch (error) {
                    // Process might already be dead
                }
            }

            model.status = 'stopped';
            model.pid = null;
            model.start_time = null;
            model.uptime_seconds = 0;

            return { success: true, message: `Model ${name} stopped` };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async checkModelHealth(port) {
        try {
            const response = await axios.get(`http://localhost:${port}/v1/models`, {
                timeout: 5000
            });
            return response.status === 200;
        } catch (error) {
            return false;
        }
    }

    async getGPUInfo() {
        try {
            const { stdout } = await execAsync(
                'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits'
            );

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

    async getGPUProcesses() {
        try {
            const { stdout } = await execAsync(
                'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits'
            );

            const lines = stdout.trim().split('\\n');
            return lines.map(line => {
                const [pid, name, memory] = line.split(',').map(s => s.trim());
                return {
                    pid: parseInt(pid),
                    name,
                    gpuMemoryMb: parseFloat(memory)
                };
            });
        } catch (error) {
            return [];
        }
    }

    async killProcessOnPort(port) {
        try {
            const { stdout } = await execAsync(`lsof -ti :${port}`);
            if (stdout.trim()) {
                const pid = parseInt(stdout.trim());
                process.kill(pid, 'SIGTERM');
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        } catch (error) {
            // Port not in use or other error
        }
    }

    async cleanupGPU() {
        try {
            // Kill all vLLM processes
            await execAsync('pkill -f "vllm.entrypoints.openai.api_server"');

            // Wait a bit, then force kill
            setTimeout(async () => {
                try {
                    await execAsync('pkill -9 -f "vllm.entrypoints.openai.api_server"');
                } catch (error) {
                    // No processes to kill
                }
            }, 2000);

            // Update all models to stopped status
            for (const [name, model] of this.models) {
                if (model.status === 'running') {
                    model.status = 'stopped';
                    model.pid = null;
                    model.start_time = null;
                    model.uptime_seconds = 0;
                }
            }

            return { success: true, message: 'GPU cleanup initiated' };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async setHFToken(token) {
        try {
            if (token && token.trim()) {
                // Save token to .env file
                let envContent = '';
                try {
                    envContent = await fs.readFile(this.envFile, 'utf8');
                } catch (error) {
                    // File doesn't exist, create it
                }

                // Update or add HF_TOKEN
                const lines = envContent.split('\\n');
                const tokenLineIndex = lines.findIndex(line => line.startsWith('HF_TOKEN='));

                if (tokenLineIndex >= 0) {
                    lines[tokenLineIndex] = `HF_TOKEN=${token.trim()}`;
                } else {
                    lines.push(`HF_TOKEN=${token.trim()}`);
                }

                await fs.writeFile(this.envFile, lines.join('\\n'));
                process.env.HF_TOKEN = token.trim();
            } else {
                // Remove token
                try {
                    await fs.unlink(this.envFile);
                } catch (error) {
                    // File doesn't exist
                }
                delete process.env.HF_TOKEN;
            }

            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async getHFToken() {
        try {
            if (process.env.HF_TOKEN) {
                return process.env.HF_TOKEN;
            }

            if (await this.fileExists(this.envFile)) {
                const envContent = await fs.readFile(this.envFile, 'utf8');
                const lines = envContent.split('\\n');
                const tokenLine = lines.find(line => line.startsWith('HF_TOKEN='));

                if (tokenLine) {
                    return tokenLine.split('=', 2)[1];
                }
            }

            return null;
        } catch (error) {
            return null;
        }
    }

    startMonitoring() {
        // Update model statistics every 5 seconds
        this.monitoringInterval = setInterval(async () => {
            await this.updateModelStats();
        }, 5000);
    }

    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }
    }

    async updateModelStats() {
        try {
            const gpuProcesses = await this.getGPUProcesses();

            for (const [name, model] of this.models) {
                if (model.status === 'running' && model.pid) {
                    try {
                        // Check if process is still running
                        process.kill(model.pid, 0);

                        // Update uptime
                        if (model.start_time) {
                            const startTime = new Date(model.start_time);
                            model.uptime_seconds = Math.floor((new Date() - startTime) / 1000);
                        }

                        // Update GPU memory usage
                        const gpuProcess = gpuProcesses.find(p => p.pid === model.pid);
                        if (gpuProcess) {
                            model.gpu_memory_mb = gpuProcess.gpuMemoryMb;
                        }

                        // Check health
                        if (!(await this.checkModelHealth(model.port))) {
                            model.status = 'unhealthy';
                            model.last_error = 'Health check failed';
                        }
                    } catch (error) {
                        // Process died
                        model.status = 'error';
                        model.last_error = 'Process died unexpectedly';
                        model.pid = null;
                    }
                }
            }
        } catch (error) {
            console.error('Error updating model stats:', error);
        }
    }

    // Cleanup method
    destroy() {
        this.stopMonitoring();
    }
}

module.exports = VLLMManager;