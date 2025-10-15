#!/usr/bin/env python3
"""
VLLM Manager - Modern Terminal Interface
A sleek, terminal-based vLLM model management system
"""

import curses
import httpx
import json
import logging
import os
import psutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import signal
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# AUTO-INSTALLATION
# =============================================================================

def setup_vllm_environment():
    """Setup vllm environment if not already installed"""
    script_dir = Path(__file__).parent
    venv_dir = script_dir / ".venv"
    venv_python = venv_dir / "bin" / "python"

    # Check if virtual environment exists and has vllm installed
    if venv_dir.exists() and venv_python.exists():
        try:
            result = subprocess.run([str(venv_python), "-c", "import vllm; print('vllm installed')"],
                                  capture_output=True, text=True, check=True)
            if result.returncode == 0:
                print("✅ vLLM environment already set up")
                return str(venv_python)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    print("🔧 Setting up vLLM environment for the first time...")
    print("This may take a few minutes as vLLM needs to be installed...")

    try:
        # Create virtual environment
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

        # Install vllm in the virtual environment
        print("⬇️  Installing vLLM (this may take 5-15 minutes)...")
        install_cmd = [str(venv_python), "-m", "pip", "install", "vllm"]

        # Run installation with real-time output
        process = subprocess.Popen(install_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in process.stdout:
            print(f"   {line.strip()}")

        process.wait()

        if process.returncode == 0:
            print("✅ vLLM installation completed successfully!")
            return str(venv_python)
        else:
            print("❌ vLLM installation failed!")
            return None

    except Exception as e:
        print(f"❌ Error setting up vLLM environment: {e}")
        return None

# Define script directory for use throughout the application
script_dir = Path(__file__).parent

# Ensure vllm environment is set up
VLLM_PYTHON = setup_vllm_environment()
if not VLLM_PYTHON:
    print("❌ Failed to setup vLLM environment. Exiting.")
    sys.exit(1)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path.home() / ".vllm-manager" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "vllm-manager.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ModelStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    UNHEALTHY = "unhealthy"

@dataclass
class ModelConfig:
    name: str
    huggingface_id: str
    port: int
    priority: int = 3
    gpu_memory_utilization: float = 0.3
    max_model_len: int = 2048
    tensor_parallel_size: int = 1

@dataclass
class ModelState:
    name: str
    config: ModelConfig
    status: ModelStatus = ModelStatus.STOPPED
    pid: Optional[int] = None
    gpu_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    requests_count: int = 0
    uptime_seconds: float = 0.0
    start_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    last_error: Optional[str] = None
    restart_count: int = 0

@dataclass
class GPUInfo:
    gpu_id: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    utilization_percent: float
    temperature: Optional[int] = None

@dataclass
class ProcessInfo:
    pid: int
    name: str
    gpu_memory_mb: float
    cpu_percent: float
    priority: int = 5

# =============================================================================
# SYSTEM MONITORING
# =============================================================================

class SystemMonitor:
    @staticmethod
    def get_gpu_info() -> List[GPUInfo]:
        """Get GPU information using nvidia-smi"""
        try:
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, check=True)

            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    gpus.append(GPUInfo(
                        gpu_id=int(parts[0]),
                        name=parts[1],
                        memory_used_mb=float(parts[2]),
                        memory_total_mb=float(parts[3]),
                        utilization_percent=float(parts[4]),
                        temperature=int(parts[5]) if parts[5] != '[Not Supported]' else None
                    ))
            return gpus
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return []

    @staticmethod
    def get_gpu_processes() -> List[ProcessInfo]:
        """Get processes using GPU"""
        try:
            result = subprocess.run([
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, check=True)

            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    pid = int(parts[0])

                    cpu_percent = 0.0
                    try:
                        process = psutil.Process(pid)
                        cpu_percent = process.cpu_percent()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                    processes.append(ProcessInfo(
                        pid=pid,
                        name=parts[1],
                        gpu_memory_mb=float(parts[2]),
                        cpu_percent=cpu_percent
                    ))
            return processes
        except Exception:
            return []

    @staticmethod
    def kill_process(pid: int, use_sudo: bool = False) -> bool:
        """Kill a process by PID, optionally with sudo"""
        try:
            logger.info(f"Attempting to kill process {pid} (sudo: {use_sudo})")

            # Check if process exists first
            try:
                process = psutil.Process(pid)
                process_name = process.name()
                logger.info(f"Found process {pid}: {process_name}")
            except psutil.NoSuchProcess:
                logger.warning(f"Process {pid} not found")
                return False

            if use_sudo:
                result = subprocess.run(["sudo", "kill", "-15", str(pid)],
                                      capture_output=True, text=True)
                time.sleep(1)
                subprocess.run(["sudo", "kill", "-9", str(pid)],
                             capture_output=True, text=True)
                success = result.returncode == 0
            else:
                os.kill(pid, 15)  # SIGTERM
                time.sleep(1)
                try:
                    os.kill(pid, 9)  # SIGKILL
                except ProcessLookupError:
                    pass
                success = True

            if success:
                logger.info(f"Successfully killed process {pid}")
            else:
                logger.error(f"Failed to kill process {pid}")
            return success

        except Exception as e:
            logger.error(f"Error killing process {pid}: {e}")
            return False

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """Check if port is in use"""
        try:
            result = subprocess.run(["lsof", "-ti", f":{port}"],
                                  capture_output=True, text=True)
            return bool(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

class ModelManager:
    def __init__(self):
        self.config_dir = Path.home() / ".vllm-manager"
        self.config_file = self.config_dir / "models.json"
        self.models: Dict[str, ModelState] = {}
        self.system_monitor = SystemMonitor()
        self.health_check_interval = 5  # Reduced from 30 for better responsiveness
        self.max_restarts = 3
        self.monitoring = False
        self.monitor_thread = None

        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.load_config()
        logger.info("ModelManager initialized")

    def load_config(self):
        """Load model configurations from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = json.load(f)
                    for model_data in data.get("models", []):
                        config = ModelConfig(**model_data)
                        self.models[config.name] = ModelState(
                            name=config.name,
                            config=config
                        )
                logger.info(f"Loaded {len(self.models)} models from config")
            except Exception as e:
                logger.error(f"Error loading config: {e}")

    def save_config(self):
        """Save model configurations to file"""
        try:
            data = {
                "models": [asdict(state.config) for state in self.models.values()]
            }
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def add_model(self, config: ModelConfig) -> bool:
        """Add a new model configuration"""
        if config.name in self.models:
            logger.warning(f"Model {config.name} already exists")
            return False

        self.models[config.name] = ModelState(name=config.name, config=config)
        self.save_config()
        logger.info(f"Added model {config.name} with config: {config}")
        return True

    def remove_model(self, name: str) -> bool:
        """Remove a model configuration"""
        if name not in self.models:
            logger.warning(f"Model {name} not found for removal")
            return False

        if self.models[name].status == ModelStatus.RUNNING:
            logger.info(f"Stopping model {name} before removal")
            self.stop_model(name)

        del self.models[name]
        self.save_config()
        logger.info(f"Removed model {name}")
        return True

    def start_model(self, name: str) -> Tuple[bool, str]:
        """Start a model"""
        if name not in self.models:
            return False, "Model not found"

        model = self.models[name]
        if model.status == ModelStatus.RUNNING:
            return False, "Model already running"

        # Check GPU availability
        gpus = self.system_monitor.get_gpu_info()
        if not gpus:
            return False, "No GPU available"

        gpu = gpus[0]
        required_memory = model.config.gpu_memory_utilization * gpu.memory_total_mb
        available_memory = gpu.memory_total_mb - gpu.memory_used_mb

        if available_memory < required_memory:
            if not self._free_gpu_memory(required_memory):
                return False, f"Insufficient GPU memory (need {required_memory:.0f}MB, have {available_memory:.0f}MB)"

        # Kill any process using the target port
        if self.system_monitor.is_port_in_use(model.config.port):
            result = subprocess.run(["lsof", "-ti", f":{model.config.port}"],
                                  capture_output=True, text=True)
            if result.stdout.strip():
                try:
                    pid = int(result.stdout.strip())
                    self.system_monitor.kill_process(pid)
                    time.sleep(1)
                except (ValueError, ProcessLookupError):
                    pass

        model.status = ModelStatus.STARTING
        logger.info(f"Starting model {name}...")

        # Use the auto-configured vllm python path
        global VLLM_PYTHON
        if not VLLM_PYTHON or not Path(VLLM_PYTHON).exists():
            return False, "VLLM Python environment not found. Please run the application again."

        cmd = [
            VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model.config.huggingface_id,
            "--host", "0.0.0.0",
            "--port", str(model.config.port),
            "--trust-remote-code",
            "--gpu-memory-utilization", str(model.config.gpu_memory_utilization),
            "--max-model-len", str(model.config.max_model_len),
            "--tensor-parallel-size", str(model.config.tensor_parallel_size),
            "--enforce-eager",
            "--disable-log-requests"
        ]

        try:
            # Get environment with HF_TOKEN
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                env["HF_TOKEN"] = hf_token

            # Start process with better error logging
            logger.info(f"Starting vLLM process with command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=script_dir,
                env=env,
                text=True
            )

            model.pid = process.pid
            model.start_time = datetime.now()
            logger.info(f"Process started (PID: {process.pid})")

            # Wait for health check
            for attempt in range(30):
                if self._check_model_health(model):
                    model.status = ModelStatus.RUNNING
                    logger.info(f"Model {name} started successfully on port {model.config.port}")
                    return True, f"Model {name} started successfully on port {model.config.port}"

                if process.poll() is not None:
                    stdout, _ = process.communicate()
                    error_msg = stdout[-1000:] if stdout else "Unknown error"
                    logger.error(f"Model {name} process failed with output: {error_msg}")
                    model.status = ModelStatus.ERROR
                    model.pid = None
                    model.last_error = error_msg
                    return False, f"Model failed to start: {error_msg}"

                time.sleep(2)

            # Timeout
            self.system_monitor.kill_process(process.pid)
            model.status = ModelStatus.ERROR
            model.pid = None
            model.last_error = "Timeout"
            return False, "Model failed to start within timeout (60 seconds)"

        except Exception as e:
            model.status = ModelStatus.ERROR
            model.pid = None
            model.last_error = str(e)
            return False, f"Failed to start model: {str(e)}"

    def stop_model(self, name: str) -> Tuple[bool, str]:
        """Stop a model"""
        if name not in self.models:
            return False, "Model not found"

        model = self.models[name]
        if model.status == ModelStatus.STOPPED:
            return False, "Model already stopped"

        if model.pid:
            if self.system_monitor.kill_process(model.pid):
                model.status = ModelStatus.STOPPED
                model.pid = None
                model.start_time = None
                logger.info(f"Model {name} stopped")
                return True, f"Model {name} stopped"
            else:
                return False, "Failed to stop model process"

        model.status = ModelStatus.STOPPED
        return True, f"Model {name} marked as stopped"

    def update_model_stats(self):
        """Update statistics for all running models"""
        gpu_processes = self.system_monitor.get_gpu_processes()

        for model in self.models.values():
            if model.status == ModelStatus.RUNNING and model.pid:
                try:
                    process = psutil.Process(model.pid)
                    if not process.is_running():
                        logger.warning(f"Model {model.name} process {model.pid} died unexpectedly")
                        model.status = ModelStatus.STOPPED
                        model.pid = None
                        model.last_error = "Process died unexpectedly"
                        continue

                    model.cpu_percent = process.cpu_percent()

                    for gpu_proc in gpu_processes:
                        if gpu_proc.pid == model.pid:
                            model.gpu_memory_mb = gpu_proc.gpu_memory_mb
                            break

                    if model.start_time:
                        model.uptime_seconds = (datetime.now() - model.start_time).total_seconds()

                    if not self._check_model_health(model):
                        logger.warning(f"Model {model.name} health check failed")
                        model.status = ModelStatus.UNHEALTHY
                    else:
                        model.last_health_check = datetime.now()

                except psutil.NoSuchProcess:
                    logger.error(f"Model {model.name} process {model.pid} not found")
                    model.status = ModelStatus.STOPPED
                    model.pid = None
                    model.last_error = "Process not found"
                except Exception as e:
                    logger.error(f"Error updating stats for model {model.name}: {e}")
                    model.status = ModelStatus.ERROR
                    model.last_error = str(e)

    def _check_model_health(self, model: ModelState) -> bool:
        """Check if model is healthy"""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"http://localhost:{model.config.port}/v1/models")
                healthy = response.status_code == 200
                if healthy:
                    logger.debug(f"Model {model.name} health check passed")
                else:
                    logger.warning(f"Model {model.name} health check returned status {response.status_code}")
                return healthy
        except Exception as e:
            logger.debug(f"Model {model.name} health check failed: {e}")
            return False

    def _free_gpu_memory(self, required_mb: float) -> bool:
        """Free GPU memory by killing lower priority processes"""
        processes = self.system_monitor.get_gpu_processes()

        if not processes:
            logger.info("No GPU processes to kill")
            return False

        processes.sort(key=lambda p: p.priority, reverse=True)

        total_freed = 0
        for process in processes:
            if process.priority >= 3:
                logger.info(f"Killing process {process.pid} ({process.name}) to free GPU memory")
                if self.system_monitor.kill_process(process.pid):
                    total_freed += process.gpu_memory_mb
                    time.sleep(2)

                    gpus = self.system_monitor.get_gpu_info()
                    if gpus:
                        available = gpus[0].memory_total_mb - gpus[0].memory_used_mb
                        if available >= required_mb:
                            logger.info(f"Freed {total_freed:.0f}MB GPU memory, now have {available:.0f}MB available")
                            return True

        logger.warning(f"Could not free enough GPU memory. Needed {required_mb:.0f}MB, freed {total_freed:.0f}MB")
        return False

    def aggressive_gpu_cleanup(self) -> Tuple[bool, str]:
        """Aggressively clean up GPU VRAM"""
        try:
            processes = self.system_monitor.get_gpu_processes()
            if not processes:
                return True, "No GPU processes found"

            killed_count = 0
            total_memory = sum(p.gpu_memory_mb for p in processes)

            for process in processes:
                try:
                    if self.system_monitor.kill_process(process.pid, use_sudo=True):
                        killed_count += 1
                        logger.info(f"Killed PID {process.pid} ({process.name}) - {process.gpu_memory_mb:.0f}MB")
                except Exception as e:
                    logger.error(f"Failed to kill PID {process.pid}: {e}")

            time.sleep(3)
            return True, f"Cleanup complete: killed {killed_count} processes, freed ~{total_memory:.0f}MB"

        except Exception as e:
            return False, f"Cleanup failed: {str(e)}"

    def start_monitoring(self):
        """Start continuous monitoring of models"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_models, daemon=True)
            self.monitor_thread.start()
            logger.info("Started model monitoring")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            logger.info("Stopped model monitoring")

    def _monitor_models(self):
        """Monitor models in background thread"""
        while self.monitoring:
            try:
                self.update_model_stats()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                time.sleep(5)

# =============================================================================
# MODERN TERMINAL UI
# =============================================================================

class ModernTerminalUI:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.running = True
        self.selected_index = 0
        self.current_view = "dashboard"
        self.status_message = ""
        self.status_time = None
        self.show_help = False
        self.colors = {}

    def init_colors(self):
        """Initialize modern color scheme"""
        try:
            curses.start_color()
            curses.use_default_colors()

            # Define modern color palette
            curses.init_pair(1, 46, -1)    # Bright green (running)
            curses.init_pair(2, 196, -1)    # Bright red (error)
            curses.init_pair(3, 226, -1)    # Bright yellow (starting)
            curses.init_pair(4, 21, -1)     # Bright blue (headers)
            curses.init_pair(5, 51, -1)     # Bright cyan (info)
            curses.init_pair(6, 201, -1)    # Bright magenta (highlight)
            curses.init_pair(7, 15, 18)     # White on dark blue (selected)
            curses.init_pair(8, 240, -1)    # Dark gray (borders)
            curses.init_pair(9, 34, -1)     # Green (unhealthy)
            curses.init_pair(10, 248, -1)   # Gray (text)
            curses.init_pair(11, 237, -1)   # Very dark gray (background dim)

            self.colors = {
                'running': curses.color_pair(1) | curses.A_BOLD,
                'error': curses.color_pair(2) | curses.A_BOLD,
                'starting': curses.color_pair(3) | curses.A_BOLD,
                'header': curses.color_pair(4) | curses.A_BOLD,
                'info': curses.color_pair(5),
                'highlight': curses.color_pair(6) | curses.A_BOLD,
                'selected': curses.color_pair(7) | curses.A_BOLD,
                'border': curses.color_pair(8),
                'unhealthy': curses.color_pair(9),
                'text': curses.color_pair(10),
                'normal': curses.A_NORMAL
            }
        except:
            # Fallback to basic colors if color initialization fails
            self.colors = {
                'running': curses.A_BOLD,
                'error': curses.A_BOLD,
                'starting': curses.A_BOLD,
                'header': curses.A_BOLD,
                'info': curses.A_NORMAL,
                'highlight': curses.A_BOLD,
                'selected': curses.A_REVERSE,
                'border': curses.A_NORMAL,
                'unhealthy': curses.A_NORMAL,
                'text': curses.A_NORMAL,
                'normal': curses.A_NORMAL
            }

    def draw_box(self, stdscr, x: int, y: int, width: int, height: int, title: str = ""):
        """Draw a modern box with rounded corners"""
        # Top border
        stdscr.addstr(y, x, "╭", self.colors['border'])
        stdscr.addstr(y, x + 1, "─" * (width - 2), self.colors['border'])
        stdscr.addstr(y, x + width - 1, "╮", self.colors['border'])

        # Title
        if title:
            title_text = f" {title} "
            title_x = x + (width - len(title_text)) // 2
            stdscr.addstr(y, title_x, title_text, self.colors['highlight'])

        # Side borders
        for i in range(1, height - 1):
            stdscr.addstr(y + i, x, "│", self.colors['border'])
            stdscr.addstr(y + i, x + width - 1, "│", self.colors['border'])

        # Bottom border
        stdscr.addstr(y + height - 1, x, "╰", self.colors['border'])
        stdscr.addstr(y + height - 1, x + 1, "─" * (width - 2), self.colors['border'])
        stdscr.addstr(y + height - 1, x + width - 1, "╯", self.colors['border'])

    def draw_header(self, stdscr):
        """Draw modern header with GPU info"""
        height, width = stdscr.getmaxyx()

        # Main title
        title = "🚀 VLLM MANAGER 🚀"
        stdscr.addstr(0, (width - len(title)) // 2, title, self.colors['header'])

        # GPU Info Box
        gpu_box_width = 60
        gpu_box_x = (width - gpu_box_width) // 2
        self.draw_box(stdscr, gpu_box_x, 2, gpu_box_width, 4, "GPU STATUS")

        gpus = self.model_manager.system_monitor.get_gpu_info()
        if gpus:
            gpu = gpus[0]
            gpu_info = [
                f"GPU: {gpu.name[:40]}",
                f"Memory: {gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f}MB ({gpu.memory_used_mb/gpu.memory_total_mb*100:.1f}%)",
                f"Utilization: {gpu.utilization_percent}%"
            ]
            if gpu.temperature:
                gpu_info[2] += f" | Temp: {gpu.temperature}°C"

            for i, info in enumerate(gpu_info):
                stdscr.addstr(3 + i, gpu_box_x + 2, info, self.colors['info'])

    def draw_models_table(self, stdscr, start_y: int):
        """Draw modern models table"""
        height, width = stdscr.getmaxyx()

        # Models box
        box_width = width - 4
        self.draw_box(stdscr, 2, start_y, box_width, height - start_y - 6, "MODELS")

        # Table headers
        headers = ["Model", "Status", "Port", "GPU(MB)", "CPU%", "Uptime", "Priority"]
        col_widths = [18, 10, 8, 10, 8, 10, 8]

        header_y = start_y + 2
        x = 4
        for header, width_h in zip(headers, col_widths):
            stdscr.addstr(header_y, x, header.ljust(width_h), self.colors['header'])
            x += width_h + 2

        # Separator
        stdscr.addstr(header_y + 1, 4, "─" * (sum(col_widths) + len(col_widths) * 2), self.colors['border'])

        # Model rows
        models = list(self.model_manager.models.values())
        for i, model in enumerate(models):
            row_y = header_y + 2 + i
            if row_y >= height - 8:
                break

            # Status colors
            if model.status == ModelStatus.RUNNING:
                color = self.colors['running']
                icon = "●"
            elif model.status == ModelStatus.STARTING:
                color = self.colors['starting']
                icon = "◐"
            elif model.status == ModelStatus.ERROR:
                color = self.colors['error']
                icon = "✗"
            elif model.status == ModelStatus.UNHEALTHY:
                color = self.colors['unhealthy']
                icon = "◑"
            else:
                color = self.colors['text']
                icon = "○"

            # Highlight selected row
            row_color = self.colors['selected'] if i == self.selected_index else color

            # Format uptime
            uptime_str = ""
            if model.uptime_seconds > 0:
                hours = int(model.uptime_seconds // 3600)
                minutes = int((model.uptime_seconds % 3600) // 60)
                uptime_str = f"{hours:02d}:{minutes:02d}"

            # Draw row
            x = 4
            values = [
                model.name[:16],
                f"{icon} {model.status.value}",
                str(model.config.port),
                f"{model.gpu_memory_mb:.0f}" if model.gpu_memory_mb > 0 else "-",
                f"{model.cpu_percent:.1f}" if model.cpu_percent > 0 else "-",
                uptime_str or "-",
                str(model.config.priority)
            ]

            for value, width_v in zip(values, col_widths):
                stdscr.addstr(row_y, x, str(value).ljust(width_v), row_color)
                x += width_v + 2

    def draw_controls(self, stdscr):
        """Draw modern controls"""
        height, width = stdscr.getmaxyx()

        controls = [
            "CONTROLS",
            "↑/↓ • Navigate   ENTER • Start/Stop   A • Add   S • Settings   E • Edit   D • Delete",
            "K • Kill Process   C • Cleanup   T • Token   H • Help   Q • Quit/Exit"
        ]

        control_y = height - 4
        for i, control in enumerate(controls):
            if i == 0:
                # Title
                title_x = (width - len(control)) // 2
                stdscr.addstr(control_y, title_x, control, self.colors['highlight'])
            else:
                text_x = (width - len(control)) // 2
                color = self.colors['info'] if i == 1 else self.colors['text']
                stdscr.addstr(control_y + i, text_x, control, color)

    def draw_help(self, stdscr):
        """Draw help screen"""
        height, width = stdscr.getmaxyx()

        help_text = [
            "🚀 VLLM MANAGER - HELP 🚀",
            "",
            "NAVIGATION:",
            "  ↑/↓ Arrow Keys    Navigate between models",
            "  ENTER or SPACE    Start/Stop selected model",
            "  ESC               Return to dashboard/Cancel dialog",
            "",
            "MODEL MANAGEMENT:",
            "  A                 Add new model configuration",
            "  S                 Configure selected model settings",
            "  D                 Delete selected model",
            "  K                 Kill selected model process",
            "  R                 Refresh all model statuses",
            "",
            "SYSTEM MANAGEMENT:",
            "  C                 Clean GPU memory",
            "  T                 Configure HuggingFace token",
            "",
            "ENVIRONMENT:",
            "  HF_TOKEN          Set in .env file for Hugging Face auth",
            "",
            "Press any key to return..."
        ]

        # Help box
        help_width = 70
        help_height = len(help_text) + 4
        help_x = (width - help_width) // 2
        help_y = (height - help_height) // 2

        self.draw_box(stdscr, help_x, help_y, help_width, help_height, "HELP")

        for i, line in enumerate(help_text):
            if help_y + 2 + i < help_y + help_height - 2:
                color = self.colors['highlight'] if line.endswith(":") else self.colors['text']
                text_x = help_x + (help_width - len(line)) // 2
                stdscr.addstr(help_y + 2 + i, text_x, line, color)

    def set_status(self, message: str):
        """Set status message"""
        self.status_message = message
        self.status_time = datetime.now()

    def run(self, stdscr):
        """Main UI loop"""
        try:
            curses.curs_set(0)
        except:
            pass  # Ignore cursor setting errors

        try:
            stdscr.nodelay(True)
        except:
            pass  # Ignore nodelay setting errors

        try:
            stdscr.timeout(100)  # Optimized timeout for performance
        except:
            stdscr.timeout(100)  # Fallback timeout

        self.init_colors()

        # Start monitoring
        self.model_manager.start_monitoring()

        last_refresh = 0
        last_background_draw = None  # Track when background was last drawn

        while self.running:
            current_time = time.time()

            # Refresh screen
            key = stdscr.getch()
            needs_refresh = key != -1 or current_time - last_refresh > 1.0  # Optimized refresh frequency

            if needs_refresh:
                # Only clear screen if not in dialog mode (to avoid flicker)
                if self.current_view not in ["add_model", "model_settings", "delete_confirm", "cleanup_confirm", "token_config", "quit_confirm"]:
                    stdscr.clear()
                    stdscr.bkgd(' ', self.colors['normal'])

                if self.show_help:
                    self.draw_help(stdscr)
                elif self.current_view in ["add_model", "model_settings", "model_edit", "delete_confirm", "cleanup_confirm", "token_config", "quit_confirm"]:
                    # For dialogs, only redraw background if needed
                    if last_background_draw != self.current_view:
                        self.draw_darkened_background(stdscr)
                        last_background_draw = self.current_view

                    # Draw the appropriate dialog
                    if self.current_view == "add_model":
                        self.draw_add_model_dialog(stdscr)
                    elif self.current_view == "model_settings":
                        self.draw_model_settings_dialog(stdscr)
                    elif self.current_view == "model_edit":
                        self.draw_model_edit_dialog(stdscr)
                    elif self.current_view == "delete_confirm":
                        self.draw_delete_confirmation_dialog(stdscr)
                    elif self.current_view == "cleanup_confirm":
                        self.draw_cleanup_confirmation_dialog(stdscr)
                    elif self.current_view == "token_config":
                        self.draw_token_dialog(stdscr)
                    elif self.current_view == "quit_confirm":
                        self.draw_quit_dialog(stdscr)
                else:
                    self.draw_header(stdscr)
                    self.draw_models_table(stdscr, 7)
                    self.draw_controls(stdscr)
                    last_background_draw = None  # Reset background tracking

                # Status message
                if self.status_message and self.status_time:
                    if (datetime.now() - self.status_time).total_seconds() < 3:
                        height, width = stdscr.getmaxyx()
                        stdscr.addstr(height - 1, 2, self.status_message[:width-4], self.colors['highlight'])

                stdscr.refresh()
                last_refresh = current_time

            # Handle input
            if key != -1:
                self.handle_input(key)

            # Optimized sleep time for better performance
            time.sleep(0.05)

        # Stop monitoring
        self.model_manager.stop_monitoring()

    def handle_input(self, key):
        """Handle keyboard input"""
        if self.show_help:
            # Any key returns from help
            self.show_help = False
            return

        # Handle dialog input first
        if self.current_view in ["add_model", "model_settings", "model_edit", "delete_confirm", "cleanup_confirm", "token_config", "quit_confirm"]:
            self.handle_dialog_input(key)
            return

        models = list(self.model_manager.models.values())

        if key == ord('q') or key == ord('Q'):  # Q key for quit
            self.show_quit_dialog()
        elif key == 27:  # ESC to cancel dialogs
            pass  # ESC handled elsewhere
        elif key == curses.KEY_UP and models:
            self.selected_index = max(0, self.selected_index - 1)
        elif key == curses.KEY_DOWN and models:
            self.selected_index = min(len(models) - 1, self.selected_index + 1)
        elif key == ord('\n') or key == ord(' '):  # Enter or Space
            if models and 0 <= self.selected_index < len(models):
                model = models[self.selected_index]
                if model.status == ModelStatus.RUNNING:
                    success, msg = self.model_manager.stop_model(model.name)
                else:
                    success, msg = self.model_manager.start_model(model.name)
                self.set_status(msg)
        elif key == ord('a') or key == ord('A'):
            self.show_add_model_dialog()
        elif key == ord('s') or key == ord('S'):
            if models and 0 <= self.selected_index < len(models):
                model = models[self.selected_index]
                self.show_model_settings_dialog(model.name)
        elif key == ord('e') or key == ord('E'):
            if models and 0 <= self.selected_index < len(models):
                model = models[self.selected_index]
                self.show_model_edit_dialog(model.name)
        elif key == ord('d') or key == ord('D'):
            if models and 0 <= self.selected_index < len(models):
                model = models[self.selected_index]
                self.show_delete_confirmation_dialog(model.name)
        elif key == ord('r') or key == ord('R'):
            self.set_status("Refreshed")
        elif key == ord('k') or key == ord('K'):
            if models and 0 <= self.selected_index < len(models):
                model = models[self.selected_index]
                if model.pid:
                    if self.model_manager.system_monitor.kill_process(model.pid):
                        self.set_status(f"Killed process {model.pid}")
                    else:
                        self.set_status("Failed to kill process")
        elif key == ord('c') or key == ord('C'):
            self.show_cleanup_confirmation_dialog()
        elif key == ord('h') or key == ord('H'):
            self.show_help = True
        elif key == ord('t') or key == ord('T'):
            self.show_token_dialog()

    def show_add_model_dialog(self):
        """Show add model dialog with full configuration"""
        self.current_view = "add_model"
        self.dialog_state = {
            'field': 'name',
            'name': '',
            'huggingface_id': '',
            'port': '8001',
            'priority': '3',
            'gpu_memory': '0.3',
            'max_len': '2048',
            'tensor_parallel': '1'
        }

    def show_model_settings_dialog(self, model_name: str):
        """Show model settings configuration dialog"""
        if model_name not in self.model_manager.models:
            return

        model = self.model_manager.models[model_name]
        self.current_view = "model_settings"
        self.dialog_state = {
            'model_name': model_name,
            'field': 'priority',
            'priority': str(model.config.priority),
            'gpu_memory': str(model.config.gpu_memory_utilization),
            'max_len': str(model.config.max_model_len),
            'tensor_parallel': str(model.config.tensor_parallel_size)
        }

    def show_model_edit_dialog(self, model_name: str):
        """Show model edit dialog for editing all fields including huggingface_id"""
        if model_name not in self.model_manager.models:
            return

        model = self.model_manager.models[model_name]
        self.current_view = "model_edit"
        self.dialog_state = {
            'model_name': model_name,
            'field': 'name',
            'name': model.name,
            'huggingface_id': model.config.huggingface_id,
            'port': str(model.config.port),
            'priority': str(model.config.priority),
            'gpu_memory': str(model.config.gpu_memory_utilization),
            'max_len': str(model.config.max_model_len),
            'tensor_parallel': str(model.config.tensor_parallel_size)
        }

    def show_delete_confirmation_dialog(self, model_name: str):
        """Show delete confirmation dialog"""
        self.current_view = "delete_confirm"
        self.dialog_state = {
            'model_name': model_name,
            'confirmed': False
        }

    def show_cleanup_confirmation_dialog(self):
        """Show GPU cleanup confirmation dialog"""
        self.current_view = "cleanup_confirm"
        self.dialog_state = {}

    def show_quit_dialog(self):
        """Show quit confirmation dialog"""
        self.current_view = "quit_confirm"
        self.dialog_state = {}

    def show_token_dialog(self):
        """Show HuggingFace token configuration dialog"""
        self.current_view = "token_config"
        # Read current token from environment or .env file
        current_token = ""
        env_file = Path.home() / ".vllm-manager" / ".env"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('HF_TOKEN='):
                            current_token = line.strip().split('=', 1)[1]
                            break
            except:
                pass
        else:
            current_token = os.environ.get("HF_TOKEN", "")

        self.dialog_state = {
            'token': current_token,
            'show_token': False  # Toggle to show/hide token
        }

    def draw_add_model_dialog(self, stdscr):
        """Draw add model dialog"""
        height, width = stdscr.getmaxyx()

        dialog_width = 80
        dialog_height = 20
        dialog_x = (width - dialog_width) // 2
        dialog_y = (height - dialog_height) // 2

        self.draw_box(stdscr, dialog_x, dialog_y, dialog_width, dialog_height, "ADD NEW MODEL")

        fields = [
            ('name', 'Model Name:'),
            ('huggingface_id', 'HuggingFace ID:'),
            ('port', 'Port:'),
            ('priority', 'Priority (1-5):'),
            ('gpu_memory', 'GPU Memory (0.1-0.9):'),
            ('max_len', 'Max Length:'),
            ('tensor_parallel', 'Tensor Parallel:')
        ]

        y_offset = dialog_y + 2
        for field, label in fields:
            # Label
            stdscr.addstr(y_offset, dialog_x + 4, label, self.colors['text'])

            # Input field
            input_value = self.dialog_state.get(field, '')
            if self.dialog_state['field'] == field:
                # Highlight current field
                stdscr.addstr(y_offset, dialog_x + 25, input_value.ljust(40), self.colors['selected'])
                # Show cursor
                if hasattr(stdscr, 'curs_set'):
                    stdscr.curs_set(1)
                    stdscr.move(y_offset, dialog_x + 25 + len(input_value))
            else:
                stdscr.addstr(y_offset, dialog_x + 25, input_value.ljust(40), self.colors['normal'])

            y_offset += 2

        # Instructions
        instructions = [
            "TAB: Next field    ENTER: Save    ESC: Cancel",
            "Popular models: mistralai/Mistral-7B-Instruct-v0.2, meta-llama/Llama-2-7b-chat-hf"
        ]

        y_offset = dialog_y + dialog_height - 4
        for instruction in instructions:
            stdscr.addstr(y_offset, dialog_x + (dialog_width - len(instruction)) // 2,
                         instruction, self.colors['info'])
            y_offset += 1

    def draw_model_settings_dialog(self, stdscr):
        """Draw model settings dialog"""
        height, width = stdscr.getmaxyx()

        dialog_width = 70
        dialog_height = 16
        dialog_x = (width - dialog_width) // 2
        dialog_y = (height - dialog_height) // 2

        model_name = self.dialog_state['model_name']
        self.draw_box(stdscr, dialog_x, dialog_y, dialog_width, dialog_height, f"SETTINGS: {model_name}")

        fields = [
            ('priority', 'Priority (1-5):'),
            ('gpu_memory', 'GPU Memory (0.1-0.9):'),
            ('max_len', 'Max Length:'),
            ('tensor_parallel', 'Tensor Parallel:')
        ]

        y_offset = dialog_y + 2
        for field, label in fields:
            # Label
            stdscr.addstr(y_offset, dialog_x + 4, label, self.colors['text'])

            # Input field
            input_value = self.dialog_state.get(field, '')
            if self.dialog_state['field'] == field:
                # Highlight current field
                stdscr.addstr(y_offset, dialog_x + 25, input_value.ljust(20), self.colors['selected'])
                # Show cursor
                if hasattr(stdscr, 'curs_set'):
                    stdscr.curs_set(1)
                    stdscr.move(y_offset, dialog_x + 25 + len(input_value))
            else:
                stdscr.addstr(y_offset, dialog_x + 25, input_value.ljust(20), self.colors['normal'])

            y_offset += 2

        # Instructions
        instructions = "TAB: Next field    ENTER: Save    ESC: Cancel"
        stdscr.addstr(dialog_y + dialog_height - 3, dialog_x + (dialog_width - len(instructions)) // 2,
                     instructions, self.colors['info'])

    def draw_model_edit_dialog(self, stdscr):
        """Draw model edit dialog"""
        height, width = stdscr.getmaxyx()

        dialog_width = 80
        dialog_height = 20
        dialog_x = (width - dialog_width) // 2
        dialog_y = (height - dialog_height) // 2

        model_name = self.dialog_state['model_name']
        self.draw_box(stdscr, dialog_x, dialog_y, dialog_width, dialog_height, f"EDIT MODEL: {model_name}")

        fields = [
            ('name', 'Model Name:'),
            ('huggingface_id', 'HuggingFace ID:'),
            ('port', 'Port:'),
            ('priority', 'Priority (1-5):'),
            ('gpu_memory', 'GPU Memory (0.1-0.9):'),
            ('max_len', 'Max Length:'),
            ('tensor_parallel', 'Tensor Parallel:')
        ]

        y_offset = dialog_y + 2
        for field, label in fields:
            # Label
            stdscr.addstr(y_offset, dialog_x + 4, label, self.colors['text'])

            # Input field
            input_value = self.dialog_state.get(field, '')
            if self.dialog_state['field'] == field:
                # Highlight current field
                stdscr.addstr(y_offset, dialog_x + 25, input_value.ljust(40), self.colors['selected'])
                # Show cursor
                if hasattr(stdscr, 'curs_set'):
                    stdscr.curs_set(1)
                    stdscr.move(y_offset, dialog_x + 25 + len(input_value))
            else:
                stdscr.addstr(y_offset, dialog_x + 25, input_value.ljust(40), self.colors['normal'])

            y_offset += 2

        # Instructions
        instructions = [
            "TAB: Next field    ENTER: Save    ESC: Cancel",
            "⚠️  Changing model name or HuggingFace ID requires model restart to take effect"
        ]

        y_offset = dialog_y + dialog_height - 4
        for instruction in instructions:
            color = self.colors['highlight'] if instruction.startswith("⚠️") else self.colors['info']
            stdscr.addstr(y_offset, dialog_x + (dialog_width - len(instruction)) // 2,
                         instruction, color)
            y_offset += 1

    def draw_delete_confirmation_dialog(self, stdscr):
        """Draw delete confirmation dialog"""
        height, width = stdscr.getmaxyx()

        dialog_width = 60
        dialog_height = 8
        dialog_x = (width - dialog_width) // 2
        dialog_y = (height - dialog_height) // 2

        self.draw_box(stdscr, dialog_x, dialog_y, dialog_width, dialog_height, "CONFIRM DELETE")

        model_name = self.dialog_state['model_name']
        warning = f"Are you sure you want to delete '{model_name}'?"
        stdscr.addstr(dialog_y + 3, dialog_x + (dialog_width - len(warning)) // 2,
                     warning, self.colors['error'])

        instructions = "Y: Yes, Delete    N: No, Cancel"
        stdscr.addstr(dialog_y + dialog_height - 3, dialog_x + (dialog_width - len(instructions)) // 2,
                     instructions, self.colors['info'])

    def draw_cleanup_confirmation_dialog(self, stdscr):
        """Draw GPU cleanup confirmation dialog"""
        height, width = stdscr.getmaxyx()

        dialog_width = 70
        dialog_height = 10
        dialog_x = (width - dialog_width) // 2
        dialog_y = (height - dialog_height) // 2

        self.draw_box(stdscr, dialog_x, dialog_y, dialog_width, dialog_height, "GPU CLEANUP")

        # Get current GPU processes
        processes = self.model_manager.system_monitor.get_gpu_processes()
        total_memory = sum(p.gpu_memory_mb for p in processes)

        warning = f"This will kill {len(processes)} processes using ~{total_memory:.0f}MB GPU memory."
        warning2 = "Only lower priority processes (3+) will be terminated."
        stdscr.addstr(dialog_y + 3, dialog_x + (dialog_width - len(warning)) // 2,
                     warning, self.colors['error'])
        stdscr.addstr(dialog_y + 4, dialog_x + (dialog_width - len(warning2)) // 2,
                     warning2, self.colors['text'])

        instructions = "Y: Yes, Cleanup    N: No, Cancel"
        stdscr.addstr(dialog_y + dialog_height - 3, dialog_x + (dialog_width - len(instructions)) // 2,
                     instructions, self.colors['info'])

    def draw_quit_dialog(self, stdscr):
        """Draw quit confirmation dialog"""
        height, width = stdscr.getmaxyx()

        dialog_width = 60
        dialog_height = 8
        dialog_x = (width - dialog_width) // 2
        dialog_y = (height - dialog_height) // 2

        self.draw_box(stdscr, dialog_x, dialog_y, dialog_width, dialog_height, "QUIT CONFIRMATION")

        message = "Are you sure you want to exit VLLM Manager?"
        stdscr.addstr(dialog_y + 3, dialog_x + (dialog_width - len(message)) // 2,
                     message, self.colors['highlight'])

        instructions = "Y: Yes, Quit    N: No, Cancel"
        stdscr.addstr(dialog_y + dialog_height - 3, dialog_x + (dialog_width - len(instructions)) // 2,
                     instructions, self.colors['info'])

    def draw_token_dialog(self, stdscr):
        """Draw HuggingFace token configuration dialog"""
        height, width = stdscr.getmaxyx()

        dialog_width = 80
        dialog_height = 12
        dialog_x = (width - dialog_width) // 2
        dialog_y = (height - dialog_height) // 2

        self.draw_box(stdscr, dialog_x, dialog_y, dialog_width, dialog_height, "HUGGINGFACE TOKEN")

        # Current token display
        token_display = self.dialog_state['token']
        if not self.dialog_state['show_token'] and token_display:
            # Mask the token
            if len(token_display) > 8:
                token_display = token_display[:4] + "*" * (len(token_display) - 8) + token_display[-4:]
            else:
                token_display = "*" * len(token_display)

        y_offset = dialog_y + 2

        # Instructions
        instructions = [
            "Enter your HuggingFace token for accessing private/gated models:",
            "",
            f"Current Token: {token_display or '(Not set)'}",
            "",
            "Commands:",
            "TAB: Show/Hide token    ENTER: Save    ESC: Cancel",
            "",
            "Get your token from: https://huggingface.co/settings/tokens"
        ]

        for i, instruction in enumerate(instructions):
            if i == 2:  # Token display line
                color = self.colors['highlight'] if self.dialog_state['show_token'] else self.colors['text']
                stdscr.addstr(y_offset + i, dialog_x + 4, instruction, color)
            elif i == 4:  # Commands header
                stdscr.addstr(y_offset + i, dialog_x + 4, instruction, self.colors['highlight'])
            else:
                stdscr.addstr(y_offset + i, dialog_x + 4, instruction, self.colors['text'])

        # Input field for new token
        input_y = y_offset + len(instructions) + 1
        stdscr.addstr(input_y, dialog_x + 4, "New Token: ", self.colors['text'])

        # Draw input field
        input_width = dialog_width - 16
        if self.dialog_state['show_token']:
            stdscr.addstr(input_y, dialog_x + 15, self.dialog_state['token'].ljust(input_width), self.colors['selected'])
            if hasattr(stdscr, 'curs_set'):
                stdscr.curs_set(1)
                stdscr.move(input_y, dialog_x + 15 + len(self.dialog_state['token']))
        else:
            masked_input = "*" * len(self.dialog_state['token'])
            stdscr.addstr(input_y, dialog_x + 15, masked_input.ljust(input_width), self.colors['selected'])
            if hasattr(stdscr, 'curs_set'):
                stdscr.curs_set(1)
                stdscr.move(input_y, dialog_x + 15 + len(masked_input))

    def draw_darkened_background(self, stdscr):
        """Draw a darkened background for dialogs efficiently"""
        height, width = stdscr.getmaxyx()

        try:
            # Simple efficient approach: clear with dark background
            stdscr.clear()
            stdscr.bkgd(' ', curses.color_pair(11))

            # Draw basic UI structure in dimmed colors
            # Draw a simple dimmed header
            title = "🚀 VLLM MANAGER 🚀"
            stdscr.addstr(0, (width - len(title)) // 2, title, curses.color_pair(11) | curses.A_DIM)

            # Draw minimal model info to show context
            models = list(self.model_manager.models.values())
            if models:
                info_text = f"{len(models)} models configured"
                stdscr.addstr(2, (width - len(info_text)) // 2, info_text, curses.color_pair(11) | curses.A_DIM)

            # Reset background for dialog rendering
            stdscr.bkgd(' ', self.colors['normal'])
        except curses.error:
            # Ultimate fallback
            stdscr.clear()
            stdscr.bkgd(' ', curses.color_pair(8))

    def handle_dialog_input(self, key):
        """Handle input in dialog mode"""
        if self.current_view == "add_model":
            self._handle_add_model_input(key)
        elif self.current_view == "model_settings":
            self._handle_model_settings_input(key)
        elif self.current_view == "model_edit":
            self._handle_model_edit_input(key)
        elif self.current_view == "delete_confirm":
            self._handle_delete_confirmation_input(key)
        elif self.current_view == "cleanup_confirm":
            self._handle_cleanup_confirmation_input(key)
        elif self.current_view == "token_config":
            self._handle_token_input(key)
        elif self.current_view == "quit_confirm":
            self._handle_quit_confirmation_input(key)

    def _handle_add_model_input(self, key):
        """Handle input in add model dialog"""
        if key == 27:  # ESC
            self.current_view = "dashboard"
            self.dialog_state = {}
            if hasattr(self, 'curs_set'):
                curses.curs_set(0)
        elif key == ord('\t'):
            # Tab to next field
            fields = ['name', 'huggingface_id', 'port', 'priority', 'gpu_memory', 'max_len', 'tensor_parallel']
            current_idx = fields.index(self.dialog_state['field'])
            self.dialog_state['field'] = fields[(current_idx + 1) % len(fields)]
        elif key == ord('\n'):
            # Save and create model
            try:
                config = ModelConfig(
                    name=self.dialog_state['name'],
                    huggingface_id=self.dialog_state['huggingface_id'],
                    port=int(self.dialog_state['port']),
                    priority=int(self.dialog_state['priority']),
                    gpu_memory_utilization=float(self.dialog_state['gpu_memory']),
                    max_model_len=int(self.dialog_state['max_len']),
                    tensor_parallel_size=int(self.dialog_state['tensor_parallel'])
                )

                if self.model_manager.add_model(config):
                    self.set_status(f"Model {config.name} added successfully")
                else:
                    self.set_status(f"Model {config.name} already exists")

                self.current_view = "dashboard"
                self.dialog_state = {}
                if hasattr(self, 'curs_set'):
                    curses.curs_set(0)
            except Exception as e:
                self.set_status(f"Error: {str(e)}")
        elif key == curses.KEY_BACKSPACE or key == 127:
            # Backspace
            current_field = self.dialog_state['field']
            self.dialog_state[current_field] = self.dialog_state[current_field][:-1]
        elif 32 <= key <= 126:  # Printable characters
            current_field = self.dialog_state['field']
            self.dialog_state[current_field] += chr(key)

    def _handle_model_settings_input(self, key):
        """Handle input in model settings dialog"""
        if key == 27:  # ESC
            self.current_view = "dashboard"
            self.dialog_state = {}
            if hasattr(self, 'curs_set'):
                curses.curs_set(0)
        elif key == ord('\t'):
            # Tab to next field
            fields = ['priority', 'gpu_memory', 'max_len', 'tensor_parallel']
            current_idx = fields.index(self.dialog_state['field'])
            self.dialog_state['field'] = fields[(current_idx + 1) % len(fields)]
        elif key == ord('\n'):
            # Save settings
            try:
                model_name = self.dialog_state['model_name']
                model = self.model_manager.models[model_name]

                # Update config
                model.config.priority = int(self.dialog_state['priority'])
                model.config.gpu_memory_utilization = float(self.dialog_state['gpu_memory'])
                model.config.max_model_len = int(self.dialog_state['max_len'])
                model.config.tensor_parallel_size = int(self.dialog_state['tensor_parallel'])

                self.model_manager.save_config()
                self.set_status(f"Settings for {model_name} updated")

                self.current_view = "dashboard"
                self.dialog_state = {}
                if hasattr(self, 'curs_set'):
                    curses.curs_set(0)
            except Exception as e:
                self.set_status(f"Error: {str(e)}")
        elif key == curses.KEY_BACKSPACE or key == 127:
            # Backspace
            current_field = self.dialog_state['field']
            self.dialog_state[current_field] = self.dialog_state[current_field][:-1]
        elif 48 <= key <= 57 or key == 46:  # Numbers and decimal point
            current_field = self.dialog_state['field']
            self.dialog_state[current_field] += chr(key)

    def _handle_model_edit_input(self, key):
        """Handle input in model edit dialog"""
        if key == 27:  # ESC
            self.current_view = "dashboard"
            self.dialog_state = {}
            if hasattr(self, 'curs_set'):
                curses.curs_set(0)
        elif key == ord('\t'):
            # Tab to next field
            fields = ['name', 'huggingface_id', 'port', 'priority', 'gpu_memory', 'max_len', 'tensor_parallel']
            current_idx = fields.index(self.dialog_state['field'])
            self.dialog_state['field'] = fields[(current_idx + 1) % len(fields)]
        elif key == ord('\n'):
            # Save changes
            try:
                old_model_name = self.dialog_state['model_name']
                new_model_name = self.dialog_state['name']

                # Check if model name changed and new name already exists
                if old_model_name != new_model_name and new_model_name in self.model_manager.models:
                    self.set_status(f"Model {new_model_name} already exists")
                    return

                model = self.model_manager.models[old_model_name]

                # Stop model if it's running and critical fields changed
                needs_restart = False
                if (model.status == ModelStatus.RUNNING and
                    (model.config.huggingface_id != self.dialog_state['huggingface_id'] or
                     model.config.port != int(self.dialog_state['port']))):
                    needs_restart = True
                    success, msg = self.model_manager.stop_model(old_model_name)
                    if not success:
                        self.set_status(f"Failed to stop model for update: {msg}")
                        return

                # Update config
                model.config.name = new_model_name
                model.config.huggingface_id = self.dialog_state['huggingface_id']
                model.config.port = int(self.dialog_state['port'])
                model.config.priority = int(self.dialog_state['priority'])
                model.config.gpu_memory_utilization = float(self.dialog_state['gpu_memory'])
                model.config.max_model_len = int(self.dialog_state['max_len'])
                model.config.tensor_parallel_size = int(self.dialog_state['tensor_parallel'])

                # Handle model name change
                if old_model_name != new_model_name:
                    # Remove old model entry
                    del self.model_manager.models[old_model_name]
                    # Create new model entry
                    model.name = new_model_name
                    self.model_manager.models[new_model_name] = model
                    # Update selected index
                    models = list(self.model_manager.models.values())
                    for i, m in enumerate(models):
                        if m.name == new_model_name:
                            self.selected_index = i
                            break

                self.model_manager.save_config()

                if needs_restart:
                    self.set_status(f"Model {new_model_name} updated and restarted")
                    # Restart the model
                    success, msg = self.model_manager.start_model(new_model_name)
                    if not success:
                        self.set_status(f"Model updated but failed to restart: {msg}")
                else:
                    self.set_status(f"Model {new_model_name} updated successfully")

                self.current_view = "dashboard"
                self.dialog_state = {}
                if hasattr(self, 'curs_set'):
                    curses.curs_set(0)
            except Exception as e:
                self.set_status(f"Error: {str(e)}")
        elif key == curses.KEY_BACKSPACE or key == 127:
            # Backspace
            current_field = self.dialog_state['field']
            self.dialog_state[current_field] = self.dialog_state[current_field][:-1]
        elif 32 <= key <= 126:  # Printable characters
            current_field = self.dialog_state['field']
            self.dialog_state[current_field] += chr(key)

    def _handle_delete_confirmation_input(self, key):
        """Handle input in delete confirmation dialog"""
        model_name = self.dialog_state['model_name']

        if key == ord('y') or key == ord('Y'):
            # Confirm delete
            if self.model_manager.remove_model(model_name):
                self.set_status(f"Model {model_name} removed")
                self.selected_index = max(0, self.selected_index - 1)
            else:
                self.set_status(f"Failed to remove model {model_name}")

            self.current_view = "dashboard"
            self.dialog_state = {}
        elif key == ord('n') or key == ord('N') or key == 27:  # N or ESC
            # Cancel delete
            self.current_view = "dashboard"
            self.dialog_state = {}

    def _handle_cleanup_confirmation_input(self, key):
        """Handle input in cleanup confirmation dialog"""
        if key == ord('y') or key == ord('Y'):
            # Confirm cleanup
            success, msg = self.model_manager.aggressive_gpu_cleanup()
            self.set_status(f"🧹 {msg}")
            self.current_view = "dashboard"
            self.dialog_state = {}
        elif key == ord('n') or key == ord('N') or key == 27:  # N or ESC
            # Cancel cleanup
            self.current_view = "dashboard"
            self.dialog_state = {}

    def _handle_token_input(self, key):
        """Handle input in token configuration dialog"""
        if key == 27:  # ESC
            self.current_view = "dashboard"
            self.dialog_state = {}
            if hasattr(self, 'curs_set'):
                curses.curs_set(0)
        elif key == ord('\t'):  # TAB - toggle show/hide token
            self.dialog_state['show_token'] = not self.dialog_state['show_token']
        elif key == ord('\n'):  # ENTER - save token
            try:
                token = self.dialog_state['token']
                if token:
                    # Save to .env file
                    env_dir = Path.home() / ".vllm-manager"
                    env_dir.mkdir(parents=True, exist_ok=True)
                    env_file = env_dir / ".env"

                    with open(env_file, 'w') as f:
                        f.write(f"HF_TOKEN={token}\n")

                    # Set in current environment
                    os.environ['HF_TOKEN'] = token

                    self.set_status("✅ HuggingFace token saved successfully")
                else:
                    # Remove token if empty
                    env_file = Path.home() / ".vllm-manager" / ".env"
                    if env_file.exists():
                        env_file.unlink()
                    if 'HF_TOKEN' in os.environ:
                        del os.environ['HF_TOKEN']

                    self.set_status("✅ HuggingFace token removed")

                self.current_view = "dashboard"
                self.dialog_state = {}
                if hasattr(self, 'curs_set'):
                    curses.curs_set(0)
            except Exception as e:
                self.set_status(f"❌ Error saving token: {str(e)}")
        elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
            # Remove last character from token
            self.dialog_state['token'] = self.dialog_state['token'][:-1]
        elif 32 <= key <= 126:  # Printable characters
            # Add character to token
            self.dialog_state['token'] += chr(key)

    def _handle_quit_confirmation_input(self, key):
        """Handle input in quit confirmation dialog"""
        if key == ord('y') or key == ord('Y'):
            # Confirm quit
            self.running = False
        elif key == ord('n') or key == ord('N') or key == 27:  # N or ESC
            # Cancel quit
            self.current_view = "dashboard"
            self.dialog_state = {}

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def launch_gui():
    """Launch the GUI terminal interface"""
    # Check if we're in an interactive terminal
    if not sys.stdin.isatty() or os.environ.get('TERM') == 'dumb':
        print("❌ VLLM Manager requires an interactive terminal to run")
        print("💡 Try running in a proper terminal (not in a script or pipe)")
        print("💡 If using SSH, ensure you have a proper TTY allocated")
        return 1

    # Initialize GUI
    model_manager = ModelManager()
    ui = ModernTerminalUI(model_manager)

    try:
        # Try to set up terminal properly before calling curses.wrapper
        os.environ.setdefault('TERM', 'xterm-256color')
        curses.wrapper(ui.run)
    except curses.error as e:
        print(f"❌ Terminal error: {e}")
        print("💡 Make sure you're running in a proper terminal environment")
        print("💡 Try: export TERM=xterm-256color")
        return 1
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    finally:
        print("👋 VLLM Manager terminated")

    return 0

def launch_nodejs_backend():
    """Launch Node.js backend and client for better performance"""
    import shutil

    # Check if Node.js is installed
    if not shutil.which("node"):
        print("❌ Node.js is not installed. Please install Node.js first.")
        print("💡 On Ubuntu: sudo apt update && sudo apt install nodejs npm")
        return 1

    # Check if nodejs-backend directory exists
    backend_dir = script_dir / "nodejs-backend"
    if not backend_dir.exists():
        print("❌ Node.js backend directory not found")
        print("💡 The Node.js backend should be in the 'nodejs-backend' directory")
        return 1

    print("🚀 Starting VLLM Manager with Node.js Backend...")
    print("📡 Starting backend server...")

    try:
        # Start the backend server
        backend_process = subprocess.Popen(
            ["node", "src/server.js"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Wait a moment for the server to start
        import time
        time.sleep(2)

        # Check if backend started successfully
        if backend_process.poll() is not None:
            stdout, _ = backend_process.communicate()
            print(f"❌ Backend failed to start: {stdout}")
            return 1

        print("✅ Backend server started successfully")
        print("🔌 Starting client...")

        # Start the client
        client_process = subprocess.Popen(
            ["node", "src/client.js"],
            cwd=backend_dir,
            preexec_fn=os.setsid  # Create new process group
        )

        # Wait for the client to finish
        try:
            client_process.wait()
        except KeyboardInterrupt:
            # Kill the entire process group
            os.killpg(os.getpgid(client_process.pid), signal.SIGTERM)

        # Clean up backend
        backend_process.terminate()
        backend_process.wait()

        print("👋 VLLM Manager (Node.js Backend) terminated")
        return 0

    except FileNotFoundError:
        print("❌ Node.js not found. Please install Node.js and npm")
        return 1
    except Exception as e:
        print(f"❌ Error starting Node.js backend: {e}")
        return 1

def handle_cli_command():
    """Handle command line operations without GUI"""
    model_manager = ModelManager()

    command = sys.argv[1].lower()

    if command == "gui":
        # Launch the GUI interface
        return launch_gui()
    elif command == "nodejs" or command == "backend":
        # Launch Node.js backend and client
        return launch_nodejs_backend()
    elif command == "start":
        if len(sys.argv) < 3:
            print("❌ Usage: vm start <model_name>")
            return 1
        model_name = sys.argv[2]
        print(f"🚀 Starting model: {model_name}")
        success, message = model_manager.start_model(model_name)
        if success:
            print(f"✅ {message}")
            return 0
        else:
            print(f"❌ {message}")
            return 1

    elif command == "stop":
        if len(sys.argv) < 3:
            print("❌ Usage: vm stop <model_name>")
            return 1
        model_name = sys.argv[2]
        print(f"🛑 Stopping model: {model_name}")
        success, message = model_manager.stop_model(model_name)
        if success:
            print(f"✅ {message}")
            return 0
        else:
            print(f"❌ {message}")
            return 1

    elif command == "list":
        print("📋 Configured Models:")
        print("-" * 60)
        for model in model_manager.models.values():
            status_icon = {
                ModelStatus.RUNNING: "●",
                ModelStatus.STARTING: "◐",
                ModelStatus.STOPPED: "○",
                ModelStatus.ERROR: "✗",
                ModelStatus.UNHEALTHY: "◑"
            }.get(model.status, "?")
            print(f"{status_icon} {model.name}")
            print(f"   HuggingFace ID: {model.config.huggingface_id}")
            print(f"   Port: {model.config.port}")
            print(f"   Status: {model.status.value}")
            if model.last_error:
                print(f"   Error: {model.last_error}")
            print()
        return 0

    elif command == "add":
        if len(sys.argv) < 4:
            print("❌ Usage: vm add <name> <huggingface_id> [port]")
            return 1
        model_name = sys.argv[2]
        hf_id = sys.argv[3]
        port = int(sys.argv[4]) if len(sys.argv) > 4 else 8001

        config = ModelConfig(
            name=model_name,
            huggingface_id=hf_id,
            port=port
        )

        if model_manager.add_model(config):
            print(f"✅ Model {model_name} added successfully")
            return 0
        else:
            print(f"❌ Model {model_name} already exists")
            return 1

    elif command == "remove":
        if len(sys.argv) < 3:
            print("❌ Usage: vm remove <model_name>")
            return 1
        model_name = sys.argv[2]
        if model_manager.remove_model(model_name):
            print(f"✅ Model {model_name} removed successfully")
            return 0
        else:
            print(f"❌ Failed to remove model {model_name}")
            return 1

    elif command == "status":
        print("🖥️  System Status:")
        print("-" * 40)

        # GPU info
        gpus = model_manager.system_monitor.get_gpu_info()
        if gpus:
            gpu = gpus[0]
            print(f"GPU: {gpu.name}")
            print(f"Memory: {gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f}MB ({gpu.memory_used_mb/gpu.memory_total_mb*100:.1f}%)")
            print(f"Utilization: {gpu.utilization_percent}%")
            if gpu.temperature:
                print(f"Temperature: {gpu.temperature}°C")
        else:
            print("❌ No GPU detected")

        print()

        # Model processes
        processes = model_manager.system_monitor.get_gpu_processes()
        if processes:
            print("🔄 Running GPU Processes:")
            for proc in processes:
                print(f"  PID {proc.pid}: {proc.name} ({proc.gpu_memory_mb:.0f}MB)")
        else:
            print("💤 No GPU processes running")
        return 0

    elif command == "cleanup":
        print("🧹 Cleaning GPU memory...")
        success, message = model_manager.aggressive_gpu_cleanup()
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
        return 0

    else:
        print(f"❌ Unknown command: {command}")
        print("💡 Available commands:")
        print("   vm gui        - Launch terminal GUI (Python)")
        print("   vm nodejs     - Launch with Node.js backend (faster)")
        print("   vm start <model> - Start a model")
        print("   vm stop <model>  - Stop a model")
        print("   vm list       - List all models")
        print("   vm add <name> <hf_id> [port] - Add model")
        print("   vm remove <model> - Remove model")
        print("   vm status     - Show system status")
        print("   vm cleanup    - Clean GPU memory")
        print()
        print("🚀 RECOMMENDED: Try the new Node.js Edition!")
        print("   💫 Modern terminal UI with better performance")
        print("   📦 Install with: curl -sSL https://your-domain.com/install-nodejs.sh | bash")
        print("   📖 Or check: README-NODEJS.md")
        return 1

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    def signal_handler(sig, frame):
        print("\n👋 Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Handle command line arguments
    if len(sys.argv) > 1:
        return handle_cli_command()

    # Default behavior: launch GUI if no arguments provided
    return launch_gui()

if __name__ == "__main__":
    main()