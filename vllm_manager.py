#!/usr/bin/env python3
"""
VLLM Manager - Modern Terminal Interface
A sleek, terminal-based vLLM model management system
"""

import curses
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
                    except:
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
        except:
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
        self.health_check_interval = 30
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
                except:
                    pass

        model.status = ModelStatus.STARTING
        logger.info(f"Starting model {name}...")

        # Get venv python path
        script_dir = Path(__file__).parent
        venv_python = script_dir / ".venv" / "bin" / "python"

        if not venv_python.exists():
            return False, f"Virtual environment Python not found at {venv_python}"

        cmd = [
            str(venv_python), "-m", "vllm.entrypoints.openai.api_server",
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

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
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
                    _, stderr = process.communicate()
                    model.status = ModelStatus.ERROR
                    model.pid = None
                    model.last_error = stderr[-500:] if stderr else "Unknown error"
                    return False, f"Model failed to start: {model.last_error}"

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
            import httpx
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"http://localhost:{model.config.port}/health")
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
            "↑/↓ • Navigate   ENTER • Start/Stop   A • Add   D • Delete   R • Refresh",
            "K • Kill Process   C • Cleanup   H • Help   Q • Quit"
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
            "  ESC               Return to dashboard",
            "",
            "MODEL MANAGEMENT:",
            "  A                 Add new model configuration",
            "  D                 Delete selected model",
            "  K                 Kill selected model process",
            "  R                 Refresh all model statuses",
            "",
            "SYSTEM MANAGEMENT:",
            "  C                 Clean GPU memory",
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
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(100)

        self.init_colors()

        # Start monitoring
        self.model_manager.start_monitoring()

        last_refresh = 0

        while self.running:
            current_time = time.time()

            # Refresh screen
            key = stdscr.getch()
            if key != -1 or current_time - last_refresh > 1:
                stdscr.clear()
                stdscr.bkgd(' ', self.colors['normal'])

                if self.show_help:
                    self.draw_help(stdscr)
                else:
                    self.draw_header(stdscr)
                    self.draw_models_table(stdscr, 7)
                    self.draw_controls(stdscr)

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

            time.sleep(0.05)

        # Stop monitoring
        self.model_manager.stop_monitoring()

    def handle_input(self, key):
        """Handle keyboard input"""
        if self.show_help:
            # Any key returns from help
            self.show_help = False
            return

        models = list(self.model_manager.models.values())

        if key == ord('q') or key == 27:  # Q or ESC
            self.running = False
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
        elif key == ord('d') or key == ord('D'):
            if models and 0 <= self.selected_index < len(models):
                model = models[self.selected_index]
                if self.model_manager.remove_model(model.name):
                    self.set_status(f"Model {model.name} removed")
                    self.selected_index = max(0, self.selected_index - 1)
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
            success, msg = self.model_manager.aggressive_gpu_cleanup()
            self.set_status(f"🧹 {msg}")
        elif key == ord('h') or key == ord('H'):
            self.show_help = True

    def show_add_model_dialog(self):
        """Show add model dialog (simplified for now)"""
        self.set_status("Add model dialog - Use CLI: vm add <name> <model_id> --port <port>")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    def signal_handler(sig, frame):
        print("\n👋 Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Initialize
    model_manager = ModelManager()
    ui = ModernTerminalUI(model_manager)

    try:
        curses.wrapper(ui.run)
    except KeyboardInterrupt:
        pass

    print("👋 VLLM Manager terminated")

if __name__ == "__main__":
    main()