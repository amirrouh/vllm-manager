const blessed = require('blessed');
const contrib = require('blessed-contrib');
const chalk = require('chalk');
const figlet = require('figlet');
const gradient = require('gradient-string');
const inquirer = require('inquirer');
const ora = require('ora');
const { spawn } = require('child_process');
const VLLMManager = require('./manager');

class ModernTerminalUI {
    constructor(manager) {
        this.manager = manager;
        this.screen = null;
        this.grid = null;
        this.modelsList = null;
        this.gpuInfo = null;
        this.systemInfo = null;
        this.controls = null;
        this.logBox = null;
        this.selectedModelIndex = 0;
        this.models = [];
        this.updating = false;
        this.updateInterval = null;
        this.wsConnection = null;
    }

    async start() {
        await this.setupScreen();
        await this.setupWebSocket();
        this.setupEventHandlers();
        this.startMonitoring();
        this.startUpdateLoop();

        this.screen.render();

        return new Promise((resolve) => {
            this.screen.key(['escape', 'q', 'Q', 'C-c'], () => {
                this.cleanup();
                resolve();
            });
        });
    }

    setupScreen() {
        // Create screen
        this.screen = blessed.screen({
            smartCSR: true,
            title: '🚀 VLLM Manager - High Performance Terminal UI',
            cursor: {
                artificial: true,
                shape: 'line',
                blink: true
            },
            debug: false,
            dump: false,
            fullUnicode: true,
            autoPadding: true,
            warnings: true
        });

        // Create grid layout
        this.grid = new contrib.grid({
            rows: 12,
            cols: 12,
            screen: this.screen,
            hideBorder: false
        });

        // Header with logo
        const headerBox = blessed.box({
            parent: this.screen,
            top: 0,
            left: 0,
            width: '100%',
            height: 3,
            content: gradient.cristal(figlet.textSync('VLLM MANAGER', {
                font: 'ANSI Shadow',
                horizontalLayout: 'default',
                verticalLayout: 'default'
            })),
            tags: true,
            style: {
                fg: 'cyan',
                bg: 'black',
                border: { fg: 'blue' }
            }
        });

        // Models table (left side)
        this.modelsList = contrib.table({
            parent: this.screen,
            label: '🤖 Models',
            border: { type: 'line', fg: 'cyan' },
            style: {
                border: { fg: 'cyan' },
                header: { fg: 'cyan', bold: true },
                cell: { fg: 'white', selected: { bg: 'blue', fg: 'white' } }
            },
            align: 'center',
            width: '70%',
            height: '50%',
            top: 4,
            left: 0,
            keys: true,
            vi: true,
            mouse: true,
            interactive: true,
            tags: true,
            columnSpacing: 2,
            columnWidth: [20, 12, 8, 10, 8, 10, 8]
        });

        // GPU Information (right side)
        this.gpuInfo = blessed.box({
            parent: this.screen,
            label: '🎮 GPU Status',
            border: { type: 'line', fg: 'green' },
            style: {
                border: { fg: 'green' },
                fg: 'white',
                bg: 'black'
            },
            width: '30%',
            height: '50%',
            top: 4,
            right: 0,
            content: '{center}Loading GPU info...{/center}',
            tags: true
        });

        // System Information (bottom left)
        this.systemInfo = blessed.box({
            parent: this.screen,
            label: '📊 System Info',
            border: { type: 'line', fg: 'magenta' },
            style: {
                border: { fg: 'magenta' },
                fg: 'white',
                bg: 'black'
            },
            width: '50%',
            height: '30%',
            bottom: 3,
            left: 0,
            content: '{center}Loading system info...{/center}',
            tags: true
        });

        // Controls help (bottom right)
        this.controls = blessed.box({
            parent: this.screen,
            label: '🎮 Controls',
            border: { type: 'line', fg: 'yellow' },
            style: {
                border: { fg: 'yellow' },
                fg: 'white',
                bg: 'black'
            },
            width: '50%',
            height: '30%',
            bottom: 3,
            right: 0,
            content: this.getControlsHelp(),
            tags: true
        });

        // Status bar at bottom
        this.statusBar = blessed.box({
            parent: this.screen,
            bottom: 0,
            left: 0,
            width: '100%',
            height: 3,
            content: '{center}{green-fg}Ready{/green-fg} | Press ? for help | ESC/Q to quit{/center}',
            tags: true,
            style: {
                fg: 'white',
                bg: 'blue',
                border: { fg: 'blue' }
            }
        });

        this.setupModelsListEvents();
    }

    setupModelsListEvents() {
        this.modelsList.focus();

        this.modelsList.on('select', (item) => {
            if (item) {
                const modelIndex = this.modelsList.selected;
                if (modelIndex >= 0 && modelIndex < this.models.length) {
                    this.showModelActions(this.models[modelIndex]);
                }
            }
        });

        this.modelsList.key(['up', 'k'], () => {
            this.modelsList.up();
            this.selectedModelIndex = Math.max(0, this.modelsList.selected);
            this.screen.render();
        });

        this.modelsList.key(['down', 'j'], () => {
            this.modelsList.down();
            this.selectedModelIndex = Math.min(this.models.length - 1, this.modelsList.selected);
            this.screen.render();
        });

        this.modelsList.key(['space', 'enter'], () => {
            const model = this.models[this.selectedModelIndex];
            if (model) {
                this.toggleModel(model);
            }
        });
    }

    setupEventHandlers() {
        // Global keyboard shortcuts
        this.screen.key(['a', 'A'], () => {
            this.showAddModelDialog();
        });

        this.screen.key(['e', 'E'], () => {
            if (this.models.length > 0) {
                const model = this.models[this.selectedModelIndex];
                this.showEditModelDialog(model);
            }
        });

        this.screen.key(['d', 'D'], () => {
            if (this.models.length > 0) {
                const model = this.models[this.selectedModelIndex];
                this.showDeleteModelDialog(model);
            }
        });

        this.screen.key(['c', 'C'], () => {
            this.showCleanupDialog();
        });

        this.screen.key(['t', 'T'], () => {
            this.showTokenDialog();
        });

        this.screen.key(['r', 'R'], () => {
            this.updateDisplay();
            this.setStatus('Display refreshed');
        });

        this.screen.key(['?'], () => {
            this.showHelp();
        });

        this.screen.key(['f5'], () => {
            this.updateDisplay();
        });

        // Handle window resize
        this.screen.on('resize', () => {
            this.modelsList.emit('resize');
            this.gpuInfo.emit('resize');
            this.systemInfo.emit('resize');
            this.controls.emit('resize');
            this.screen.render();
        });
    }

    getControlsHelp() {
        return [
            '{center}{bold-fg}CONTROLS{/bold-fg}{/center}',
            '{center}↑/k • Navigate down{/center}',
            '{center}↓/j • Navigate up{/center}',
            '{center}Enter • Start/Stop model{/center}',
            '{center}A • Add model{/center}',
            '{center}E • Edit selected{/center}',
            '{center}D • Delete selected{/center}',
            '{center}C • Cleanup GPU{/center}',
            '{center}T • Configure token{/center}',
            '{center}R • Refresh{/center}',
            '{center}? • Help{/center}',
            '{center}Q/ESC • Quit{/center}'
        ].join('\\n');
    }

    async setupWebSocket() {
        try {
            // Connect to WebSocket for real-time updates
            const WebSocket = require('ws');
            this.wsConnection = new WebSocket('ws://localhost:3001');

            this.wsConnection.on('open', () => {
                this.setStatus('{green-fg}Connected to backend{/green-fg}');
            });

            this.wsConnection.on('message', (data) => {
                const message = JSON.parse(data);
                if (message.type === 'models') {
                    this.models = message.data || [];
                    this.updateModelsDisplay();
                }
            });

            this.wsConnection.on('close', () => {
                this.setStatus('{yellow-fg}Backend connection lost{/yellow-fg}');
                // Try to reconnect after 5 seconds
                setTimeout(() => this.setupWebSocket(), 5000);
            });

            this.wsConnection.on('error', (error) => {
                this.setStatus(`{red-fg}Connection error: ${error.message}{/red-fg}`);
            });

        } catch (error) {
            this.setStatus(`{red-fg}Failed to connect to backend: ${error.message}{/red-fg}`);
        }
    }

    startMonitoring() {
        // Load initial data
        this.updateDisplay();
    }

    startUpdateLoop() {
        // Update display every 2 seconds
        this.updateInterval = setInterval(() => {
            if (!this.updating) {
                this.updateDisplay();
            }
        }, 2000);
    }

    async updateDisplay() {
        this.updating = true;

        try {
            // Get fresh data from manager
            const [models, gpuInfo, processes] = await Promise.all([
                this.manager.getAllModels(),
                this.manager.getGPUInfo(),
                this.manager.getGPUProcesses()
            ]);

            this.models = models;

            // Update displays
            this.updateModelsDisplay();
            this.updateGPUDisplay(gpuInfo);
            this.updateSystemDisplay(processes);

        } catch (error) {
            this.setStatus(`{red-fg}Update error: ${error.message}{/red-fg}`);
        } finally {
            this.updating = false;
            this.screen.render();
        }
    }

    updateModelsDisplay() {
        const data = this.models.map((model, index) => {
            const statusIcon = this.getStatusIcon(model.status);
            const statusColor = this.getStatusColor(model.status);
            const uptime = model.uptime_seconds > 0 ? this.formatUptime(model.uptime_seconds) : '-';
            const isSelected = index === this.selectedModelIndex;

            return [
                isSelected ? `{blue-fg}${model.name}{/blue-fg}` : model.name,
                statusColor(statusIcon + ' ' + model.status),
                model.port.toString(),
                model.gpu_memory_mb > 0 ? model.gpu_memory_mb.toFixed(0) : '-',
                model.cpu_percent > 0 ? model.cpu_percent.toFixed(1) + '%' : '-',
                uptime,
                model.priority.toString()
            ];
        });

        this.modelsList.setData({
            headers: ['Model', 'Status', 'Port', 'GPU(MB)', 'CPU%', 'Uptime', 'Priority'],
            data: data
        });

        // Ensure selected index is valid
        if (this.selectedModelIndex >= this.models.length) {
            this.selectedModelIndex = Math.max(0, this.models.length - 1);
            this.modelsList.select(this.selectedModelIndex);
        }
    }

    updateGPUDisplay(gpuInfo) {
        let content = [];

        if (gpuInfo && gpuInfo.length > 0) {
            const gpu = gpuInfo[0];
            content.push(`{center}{bold-fg}GPU #${gpu.index}{/bold-fg}{/center}`);
            content.push('');
            content.push(`{cyan-fg}Name:{/cyan-fg} ${gpu.name}`);
            content.push(`{cyan-fg}Memory:{/cyan-fg} ${gpu.memoryUsed.toFixed(0)}/${gpu.memoryTotal.toFixed(0)}MB`);
            content.push(`{cyan-fg}Utilization:{/cyan-fg} ${gpu.utilization}%`);
            if (gpu.temperature) {
                content.push(`{cyan-fg}Temperature:{/cyan-fg} ${gpu.temperature}°C`);
            }
        } else {
            content.push('{center}{red-fg}❌ No GPU detected{/red-fg}{/center}');
        }

        this.gpuInfo.setContent(content.join('\\n'));
    }

    updateSystemDisplay(processes) {
        let content = [];

        content.push(`{center}{bold-fg}System Status{/bold-fg}{/center}`);
        content.push('');

        // Running models count
        const runningCount = this.models.filter(m => m.status === 'running').length;
        content.push(`{green-fg}Running Models:{/green-fg} ${runningCount}/${this.models.length}`);
        content.push(`{green-fg}GPU Processes:{/green-fg} ${processes.length}`);

        if (processes.length > 0) {
            content.push('');
            content.push(`{cyan-fg}Active Processes:{/cyan-fg}`);
            processes.slice(0, 3).forEach(proc => {
                content.push(`  • ${proc.name} (${proc.gpuMemoryMb.toFixed(0)}MB)`);
            });
            if (processes.length > 3) {
                content.push(`  ... and ${processes.length - 3} more`);
            }
        }

        this.systemInfo.setContent(content.join('\\n'));
    }

    async showModelActions(model) {
        const { action } = await inquirer.prompt([
            {
                type: 'list',
                name: 'action',
                message: `Select action for model "${model.name}":`,
                choices: [
                    { name: model.status === 'running' ? '⏹️  Stop Model' : '▶️  Start Model', value: 'toggle' },
                    { name: '✏️  Edit Model', value: 'edit' },
                    { name: '🗑️  Delete Model', value: 'delete' },
                    { name: '❌ Cancel', value: 'cancel' }
                ]
            }
        ]);

        switch (action) {
            case 'toggle':
                await this.toggleModel(model);
                break;
            case 'edit':
                await this.showEditModelDialog(model);
                break;
            case 'delete':
                await this.showDeleteModelDialog(model);
                break;
        }
    }

    async toggleModel(model) {
        this.setStatus(`${model.status === 'running' ? 'Stopping' : 'Starting'} model: ${model.name}`);

        try {
            let result;
            if (model.status === 'running') {
                result = await this.manager.stopModel(model.name);
            } else {
                result = await this.manager.startModel(model.name);
            }

            if (result.success) {
                this.setStatus(`{green-fg}✅ ${result.message}{/green-fg}`);
            } else {
                this.setStatus(`{red-fg}❌ ${result.error}{/red-fg}`);
            }
        } catch (error) {
            this.setStatus(`{red-fg}❌ Operation failed: ${error.message}{/red-fg}`);
        }

        // Update display after operation
        setTimeout(() => this.updateDisplay(), 1000);
    }

    async showAddModelDialog() {
        this.screen.grabInput = false;

        const questions = [
            {
                type: 'input',
                name: 'name',
                message: 'Model name:',
                validate: input => input.trim() !== '' || 'Model name is required'
            },
            {
                type: 'input',
                name: 'huggingfaceId',
                message: 'HuggingFace ID:',
                validate: input => input.trim() !== '' || 'HuggingFace ID is required'
            },
            {
                type: 'number',
                name: 'port',
                message: 'Port:',
                default: 8001,
                validate: input => input > 1024 && input < 65536 || 'Port must be between 1024 and 65536'
            },
            {
                type: 'number',
                name: 'priority',
                message: 'Priority (1-5):',
                default: 3,
                validate: input => input >= 1 && input <= 5 || 'Priority must be between 1 and 5'
            },
            {
                type: 'number',
                name: 'gpuMemory',
                message: 'GPU Memory Utilization (0.1-0.9):',
                default: 0.3,
                validate: input => input >= 0.1 && input <= 0.9 || 'GPU memory must be between 0.1 and 0.9'
            },
            {
                type: 'number',
                name: 'maxLen',
                message: 'Max Model Length:',
                default: 2048,
                validate: input => input > 0 || 'Max length must be greater than 0'
            }
        ];

        try {
            const answers = await inquirer.prompt(questions);

            this.setStatus('Adding model...');

            const result = await this.manager.addModel({
                name: answers.name,
                huggingface_id: answers.huggingfaceId,
                port: answers.port,
                priority: answers.priority,
                gpu_memory_utilization: answers.gpuMemory,
                max_model_len: answers.maxLen,
                tensor_parallel_size: 1
            });

            if (result.success) {
                this.setStatus(`{green-fg}✅ Model ${answers.name} added successfully{/green-fg}`);
            } else {
                this.setStatus(`{red-fg}❌ Failed to add model: ${result.error}{/red-fg}`);
            }
        } catch (error) {
            if (error.name !== 'ExitPromptError') {
                this.setStatus(`{red-fg}❌ Failed to add model: ${error.message}{/red-fg}`);
            }
        } finally {
            this.screen.grabInput = true;
            this.screen.focus();
            this.updateDisplay();
        }
    }

    async showEditModelDialog(model) {
        this.screen.grabInput = false;

        const questions = [
            {
                type: 'input',
                name: 'name',
                message: 'Model name:',
                default: model.name,
                validate: input => input.trim() !== '' || 'Model name is required'
            },
            {
                type: 'input',
                name: 'huggingfaceId',
                message: 'HuggingFace ID:',
                default: model.huggingface_id,
                validate: input => input.trim() !== '' || 'HuggingFace ID is required'
            },
            {
                type: 'number',
                name: 'port',
                message: 'Port:',
                default: model.port,
                validate: input => input > 1024 && input < 65536 || 'Port must be between 1024 and 65536'
            },
            {
                type: 'number',
                name: 'priority',
                message: 'Priority (1-5):',
                default: model.priority,
                validate: input => input >= 1 && input <= 5 || 'Priority must be between 1 and 5'
            },
            {
                type: 'number',
                name: 'gpuMemory',
                message: 'GPU Memory Utilization (0.1-0.9):',
                default: model.gpu_memory_utilization,
                validate: input => input >= 0.1 && input <= 0.9 || 'GPU memory must be between 0.1 and 0.9'
            },
            {
                type: 'number',
                name: 'maxLen',
                message: 'Max Model Length:',
                default: model.max_model_len,
                validate: input => input > 0 || 'Max length must be greater than 0'
            }
        ];

        try {
            const answers = await inquirer.prompt(questions);

            this.setStatus('Updating model...');

            const updateData = {};
            if (answers.name !== model.name) updateData.name = answers.name;
            if (answers.huggingfaceId !== model.huggingface_id) updateData.huggingface_id = answers.huggingfaceId;
            if (answers.port !== model.port) updateData.port = answers.port;
            if (answers.priority !== model.priority) updateData.priority = answers.priority;
            if (answers.gpuMemory !== model.gpu_memory_utilization) updateData.gpu_memory_utilization = answers.gpuMemory;
            if (answers.maxLen !== model.max_model_len) updateData.max_model_len = answers.maxLen;

            if (Object.keys(updateData).length > 0) {
                const result = await this.manager.updateModel(model.name, updateData);

                if (result.success) {
                    this.setStatus(`{green-fg}✅ Model updated successfully{/green-fg}`);
                } else {
                    this.setStatus(`{red-fg}❌ Failed to update model: ${result.error}{/red-fg}`);
                }
            } else {
                this.setStatus('No changes made');
            }
        } catch (error) {
            if (error.name !== 'ExitPromptError') {
                this.setStatus(`{red-fg}❌ Failed to update model: ${error.message}{/red-fg}`);
            }
        } finally {
            this.screen.grabInput = true;
            this.screen.focus();
            this.updateDisplay();
        }
    }

    async showDeleteModelDialog(model) {
        this.screen.grabInput = false;

        try {
            const { confirm } = await inquirer.prompt([
                {
                    type: 'confirm',
                    name: 'confirm',
                    message: `Are you sure you want to delete '${model.name}'?`,
                    default: false
                }
            ]);

            if (confirm) {
                this.setStatus('Deleting model...');

                const result = await this.manager.removeModel(model.name);

                if (result.success) {
                    this.setStatus(`{green-fg}✅ Model ${model.name} deleted successfully{/green-fg}`);
                    this.selectedModelIndex = Math.max(0, this.selectedModelIndex - 1);
                } else {
                    this.setStatus(`{red-fg}❌ Failed to delete model: ${result.error}{/red-fg}`);
                }
            } else {
                this.setStatus('Model deletion cancelled');
            }
        } catch (error) {
            if (error.name !== 'ExitPromptError') {
                this.setStatus(`{red-fg}❌ Failed to delete model: ${error.message}{/red-fg}`);
            }
        } finally {
            this.screen.grabInput = true;
            this.screen.focus();
            this.updateDisplay();
        }
    }

    async showCleanupDialog() {
        this.screen.grabInput = false;

        try {
            const { confirm } = await inquirer.prompt([
                {
                    type: 'confirm',
                    name: 'confirm',
                    message: 'This will kill all GPU processes. Are you sure?',
                    default: false
                }
            ]);

            if (confirm) {
                this.setStatus('Cleaning GPU memory...');

                const result = await this.manager.cleanupGPU();

                if (result.success) {
                    this.setStatus(`{green-fg}✅ ${result.message}{/green-fg}`);
                } else {
                    this.setStatus(`{red-fg}❌ ${result.error}{/red-fg}`);
                }
            } else {
                this.setStatus('GPU cleanup cancelled');
            }
        } catch (error) {
            if (error.name !== 'ExitPromptError') {
                this.setStatus(`{red-fg}❌ Failed to cleanup GPU: ${error.message}{/red-fg}`);
            }
        } finally {
            this.screen.grabInput = true;
            this.screen.focus();
            this.updateDisplay();
        }
    }

    async showTokenDialog() {
        this.screen.grabInput = false;

        try {
            const { token } = await inquirer.prompt([
                {
                    type: 'password',
                    name: 'token',
                    message: 'Enter your HuggingFace token:',
                    mask: '*'
                }
            ]);

            this.setStatus('Saving token...');

            await this.manager.setHFToken(token);
            this.setStatus('{green-fg}✅ HuggingFace token saved successfully{/green-fg}');

        } catch (error) {
            if (error.name !== 'ExitPromptError') {
                this.setStatus(`{red-fg}❌ Failed to save token: ${error.message}{/red-fg}`);
            }
        } finally {
            this.screen.grabInput = true;
            this.screen.focus();
        }
    }

    showHelp() {
        this.screen.grabInput = false;

        const helpText = [
            '{bold-fg}{center}🚀 VLLM MANAGER - HELP{/center}{/bold-fg}',
            '',
            '{bold-fg}NAVIGATION:{/bold-fg}',
            '  ↑/k or Down Arrow/J  Navigate between models',
            '  Enter or Space       Start/Stop selected model',
            '  ESC                  Return to UI/Cancel dialog',
            '',
            '{bold-fg}MODEL MANAGEMENT:{/bold-fg}',
            '  A                    Add new model configuration',
            '  E                    Edit selected model',
            '  D                    Delete selected model',
            '  R                    Refresh display',
            '',
            '{bold-fg}SYSTEM MANAGEMENT:{/bold-fg}',
            '  C                    Clean GPU memory',
            '  T                    Configure HuggingFace token',
            '  ?                    Show this help',
            '',
            '{bold-fg}EXIT:{/bold-fg}',
            '  Q or ESC             Quit VLLM Manager',
            '',
            '{center}{yellow-fg}Press any key to return{/yellow-fg}{/center}'
        ];

        const helpBox = blessed.box({
            parent: this.screen,
            top: 'center',
            left: 'center',
            width: '80%',
            height: '80%',
            content: helpText.join('\\n'),
            border: { type: 'line', fg: 'cyan' },
            style: {
                border: { fg: 'cyan' },
                fg: 'white',
                bg: 'black'
            },
            tags: true,
            scrollable: true
        });

        this.screen.append(helpBox);
        helpBox.focus();
        this.screen.render();

        helpBox.key(['escape', 'q', 'Q', 'enter'], () => {
            this.screen.remove(helpBox);
            this.modelsList.focus();
            this.screen.render();
        });
    }

    setStatus(message) {
        this.statusBar.setContent(`{center}${message}{/center}`);
        this.screen.render();
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

    getStatusColor(status) {
        return (text) => {
            const colors = {
                'running': chalk.green,
                'starting': chalk.yellow,
                'stopped': chalk.gray,
                'error': chalk.red,
                'unhealthy': chalk.magenta
            };
            return colors[status] ? colors[status](text) : text;
        };
    }

    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
    }

    cleanup() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        if (this.wsConnection) {
            this.wsConnection.close();
        }

        this.screen.destroy();
        console.log('\\n👋 Goodbye!');
    }
}

module.exports = ModernTerminalUI;