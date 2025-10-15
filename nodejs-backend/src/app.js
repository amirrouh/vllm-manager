#!/usr/bin/env node

const { Command } = require('commander');
const chalk = require('chalk');
const figlet = require('figlet');
const gradient = require('gradient-string');
const inquirer = require('inquirer');
const ora = require('ora');
const Table = require('cli-table3');
const ModernTerminalUI = require('./ui');
const VLLMManager = require('./manager');

const program = new Command();

// ASCII Art Logo
function showLogo() {
    console.log(
        gradient.retro(figlet.textSync('VLLM Manager', {
            font: 'ANSI Shadow',
            horizontalLayout: 'default',
            verticalLayout: 'default'
        }))
    );
    console.log(chalk.cyan('                 High-Performance Terminal UI for vLLM Models\n'));
}

// CLI Functions
async function listModels(manager) {
    const models = await manager.getAllModels();

    if (models.length === 0) {
        console.log(chalk.yellow('⚠️  No models configured. Use "add" command to add a model.'));
        return;
    }

    const table = new Table({
        head: [
            chalk.cyan('Model'),
            chalk.cyan('Status'),
            chalk.cyan('Port'),
            chalk.cyan('GPU(MB)'),
            chalk.cyan('CPU%'),
            chalk.cyan('Uptime'),
            chalk.cyan('Priority')
        ],
        colWidths: [20, 12, 8, 10, 8, 10, 8]
    });

    models.forEach(model => {
        const statusIcon = getStatusIcon(model.status);
        const statusColor = getStatusColor(model.status);
        const uptime = model.uptime_seconds > 0 ? formatUptime(model.uptime_seconds) : '-';

        table.push([
            model.name,
            statusColor(statusIcon + ' ' + model.status),
            model.port,
            model.gpu_memory_mb > 0 ? model.gpu_memory_mb.toFixed(0) : '-',
            model.cpu_percent > 0 ? model.cpu_percent.toFixed(1) + '%' : '-',
            uptime,
            model.priority
        ]);
    });

    console.log(table.toString());
}

async function startModel(manager, modelName) {
    const spinner = ora(`🚀 Starting model: ${modelName}`).start();

    try {
        const result = await manager.startModel(modelName);

        if (result.success) {
            spinner.succeed(chalk.green(`✅ ${result.message}`));
        } else {
            spinner.fail(chalk.red(`❌ ${result.error}`));
        }
    } catch (error) {
        spinner.fail(chalk.red(`❌ Failed to start model: ${error.message}`));
    }
}

async function stopModel(manager, modelName) {
    const spinner = ora(`🛑 Stopping model: ${modelName}`).start();

    try {
        const result = await manager.stopModel(modelName);

        if (result.success) {
            spinner.succeed(chalk.green(`✅ ${result.message}`));
        } else {
            spinner.fail(chalk.red(`❌ ${result.error}`));
        }
    } catch (error) {
        spinner.fail(chalk.red(`❌ Failed to stop model: ${error.message}`));
    }
}

async function addModel(manager, name, huggingfaceId, port) {
    const questions = [
        {
            type: 'input',
            name: 'name',
            message: 'Model name:',
            default: name,
            validate: input => input.trim() !== '' || 'Model name is required'
        },
        {
            type: 'input',
            name: 'huggingfaceId',
            message: 'HuggingFace ID:',
            default: huggingfaceId,
            validate: input => input.trim() !== '' || 'HuggingFace ID is required'
        },
        {
            type: 'number',
            name: 'port',
            message: 'Port:',
            default: port || 8001,
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

    const answers = await inquirer.prompt(questions);

    const spinner = ora('📦 Adding model...').start();

    try {
        const result = await manager.addModel({
            name: answers.name,
            huggingface_id: answers.huggingfaceId,
            port: answers.port,
            priority: answers.priority,
            gpu_memory_utilization: answers.gpuMemory,
            max_model_len: answers.maxLen,
            tensor_parallel_size: 1
        });

        if (result.success) {
            spinner.succeed(chalk.green(`✅ Model ${answers.name} added successfully`));
        } else {
            spinner.fail(chalk.red(`❌ Failed to add model: ${result.error}`));
        }
    } catch (error) {
        spinner.fail(chalk.red(`❌ Failed to add model: ${error.message}`));
    }
}

async function removeModel(manager, modelName) {
    const { confirm } = await inquirer.prompt([
        {
            type: 'confirm',
            name: 'confirm',
            message: `Are you sure you want to delete '${modelName}'?`,
            default: false
        }
    ]);

    if (!confirm) {
        console.log(chalk.yellow('🚫 Model deletion cancelled'));
        return;
    }

    const spinner = ora(`🗑️  Removing model: ${modelName}`).start();

    try {
        const result = await manager.removeModel(modelName);

        if (result.success) {
            spinner.succeed(chalk.green(`✅ Model ${modelName} removed successfully`));
        } else {
            spinner.fail(chalk.red(`❌ Failed to remove model: ${result.error}`));
        }
    } catch (error) {
        spinner.fail(chalk.red(`❌ Failed to remove model: ${error.message}`));
    }
}

async function showStatus(manager) {
    const spinner = ora('🔍 Getting system status...').start();

    try {
        const [models, gpuInfo, processes] = await Promise.all([
            manager.getAllModels(),
            manager.getGPUInfo(),
            manager.getGPUProcesses()
        ]);

        spinner.stop();

        console.log(chalk.bold.blue('🖥️  System Status\n'));

        // GPU Information
        console.log(chalk.bold.cyan('🎮 GPU Information:'));
        if (gpuInfo.length > 0) {
            const gpu = gpuInfo[0];
            console.log(`   Name: ${gpu.name}`);
            console.log(`   Memory: ${gpu.memoryUsed.toFixed(0)}/${gpu.memoryTotal.toFixed(0)}MB (${(gpu.memoryUsed/gpu.memoryTotal*100).toFixed(1)}%)`);
            console.log(`   Utilization: ${gpu.utilization}%`);
            if (gpu.temperature) {
                console.log(`   Temperature: ${gpu.temperature}°C`);
            }
        } else {
            console.log(chalk.red('   ❌ No GPU detected'));
        }

        console.log();

        // Running Models
        const runningModels = models.filter(m => m.status === 'running');
        console.log(chalk.bold.cyan('🚀 Running Models:'));
        if (runningModels.length > 0) {
            runningModels.forEach(model => {
                console.log(`   ● ${model.name} (Port: ${model.port}, Uptime: ${formatUptime(model.uptime_seconds)})`);
            });
        } else {
            console.log(chalk.gray('   💤 No models running'));
        }

        console.log();

        // GPU Processes
        console.log(chalk.bold.cyan('🔄 GPU Processes:'));
        if (processes.length > 0) {
            processes.forEach(proc => {
                console.log(`   PID ${proc.pid}: ${proc.name} (${proc.gpuMemoryMb.toFixed(0)}MB)`);
            });
        } else {
            console.log(chalk.gray('   💤 No GPU processes running'));
        }

    } catch (error) {
        spinner.fail(chalk.red(`❌ Failed to get status: ${error.message}`));
    }
}

async function cleanupGPU(manager) {
    const { confirm } = await inquirer.prompt([
        {
            type: 'confirm',
            name: 'confirm',
            message: 'This will kill all GPU processes. Are you sure?',
            default: false
        }
    ]);

    if (!confirm) {
        console.log(chalk.yellow('🚫 GPU cleanup cancelled'));
        return;
    }

    const spinner = ora('🧹 Cleaning GPU memory...').start();

    try {
        const result = await manager.cleanupGPU();

        if (result.success) {
            spinner.succeed(chalk.green(`✅ ${result.message}`));
        } else {
            spinner.fail(chalk.red(`❌ ${result.error}`));
        }
    } catch (error) {
        spinner.fail(chalk.red(`❌ Failed to cleanup GPU: ${error.message}`));
    }
}

// Utility functions
function getStatusIcon(status) {
    const icons = {
        'running': '●',
        'starting': '◐',
        'stopped': '○',
        'error': '✗',
        'unhealthy': '◑'
    };
    return icons[status] || '?';
}

function getStatusColor(status) {
    const colors = {
        'running': chalk.green,
        'starting': chalk.yellow,
        'stopped': chalk.gray,
        'error': chalk.red,
        'unhealthy': chalk.magenta
    };
    return colors[status] || chalk.white;
}

function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
}

// Main CLI Program
program
    .name('vllm-manager')
    .description('High-performance VLLM Manager with modern terminal UI')
    .version('1.0.0');

program
    .command('gui')
    .description('Launch the modern terminal GUI')
    .action(async () => {
        showLogo();
        const manager = new VLLMManager();
        const ui = new ModernTerminalUI(manager);

        try {
            await ui.start();
        } catch (error) {
            console.error(chalk.red(`❌ Failed to start GUI: ${error.message}`));
            process.exit(1);
        }
    });

program
    .command('list')
    .alias('ls')
    .description('List all configured models')
    .action(async () => {
        const manager = new VLLMManager();
        await listModels(manager);
    });

program
    .command('start <model>')
    .description('Start a model')
    .action(async (model) => {
        const manager = new VLLMManager();
        await startModel(manager, model);
    });

program
    .command('stop <model>')
    .description('Stop a model')
    .action(async (model) => {
        const manager = new VLLMManager();
        await stopModel(manager, model);
    });

program
    .command('add [name] [huggingface_id] [port]')
    .description('Add a new model')
    .action(async (name, huggingfaceId, port) => {
        const manager = new VLLMManager();
        await addModel(manager, name, huggingfaceId, port);
    });

program
    .command('remove <model>')
    .alias('rm')
    .description('Remove a model')
    .action(async (model) => {
        const manager = new VLLMManager();
        await removeModel(manager, model);
    });

program
    .command('status')
    .description('Show system status')
    .action(async () => {
        const manager = new VLLMManager();
        await showStatus(manager);
    });

program
    .command('cleanup')
    .description('Clean GPU memory')
    .action(async () => {
        const manager = new VLLMManager();
        await cleanupGPU(manager);
    });

program
    .command('config')
    .description('Configure HuggingFace token')
    .action(async () => {
        const { token } = await inquirer.prompt([
            {
                type: 'password',
                name: 'token',
                message: 'Enter your HuggingFace token:',
                mask: '*'
            }
        ]);

        const manager = new VLLMManager();
        const spinner = ora('💾 Saving token...').start();

        try {
            await manager.setHFToken(token);
            spinner.succeed(chalk.green('✅ HuggingFace token saved successfully'));
        } catch (error) {
            spinner.fail(chalk.red(`❌ Failed to save token: ${error.message}`));
        }
    });

// Default action - launch GUI if no command provided
program.action(() => {
    showLogo();
    console.log(chalk.yellow('ℹ️  No command specified. Launching GUI...\n'));
    console.log(chalk.gray('Use --help to see available commands\n'));

    const manager = new VLLMManager();
    const ui = new ModernTerminalUI(manager);

    ui.start().catch(error => {
        console.error(chalk.red(`❌ Failed to start GUI: ${error.message}`));
        process.exit(1);
    });
});

// Parse command line arguments
program.parse();

// If no command provided, show help
if (!process.argv.slice(2).length) {
    showLogo();
    program.outputHelp();
}