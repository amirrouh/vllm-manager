#!/usr/bin/env python3
"""
VLLM Terminal Manager - Cool Multi-Model Management Interface
Clean, organized, and OCD-friendly code structure
"""

import asyncio
import json
import os
import psutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import curses
import threading
import signal
import sys


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
    priority: int = 5  # Default to lowest priority for cleanup


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
            print(f"Error getting GPU info: {e}")
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
    def kill_process(pid: int) -> bool:
        """Kill a process by PID"""
        try:
            os.kill(pid, 15)  # SIGTERM
            time.sleep(1)
            try:
                os.kill(pid, 9)  # SIGKILL
            except ProcessLookupError:
                pass
            return True
        except Exception:
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
    def __init__(self, config_file: str = "models_config.json"):
        self.config_file = Path(config_file)
        self.models: Dict[str, ModelState] = {}
        self.system_monitor = SystemMonitor()
        self.load_config()

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
            except Exception as e:
                print(f"Error loading config: {e}")

    def save_config(self):
        """Save model configurations to file"""
        try:
            data = {
                "models": [asdict(state.config) for state in self.models.values()]
            }
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def add_model(self, config: ModelConfig) -> bool:
        """Add a new model configuration"""
        if config.name in self.models:
            return False
        
        self.models[config.name] = ModelState(name=config.name, config=config)
        self.save_config()
        return True

    def remove_model(self, name: str) -> bool:
        """Remove a model configuration"""
        if name not in self.models:
            return False
        
        # Stop model if running
        if self.models[name].status == ModelStatus.RUNNING:
            self.stop_model(name)
        
        del self.models[name]
        self.save_config()
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
            # Try to free memory by killing lower priority processes
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

        # Start the model
        model.status = ModelStatus.STARTING
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model.config.huggingface_id,
            "--host", "0.0.0.0",
            "--port", str(model.config.port),
            "--trust-remote-code",
            "--device", "cuda",
            "--gpu-memory-utilization", str(model.config.gpu_memory_utilization),
            "--max-model-len", str(model.config.max_model_len),
            "--tensor-parallel-size", str(model.config.tensor_parallel_size),
            "--enforce-eager",
            "--disable-log-requests"
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd="/home/ubuntu/apps/vllm",
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            )
            
            model.pid = process.pid
            model.start_time = datetime.now()
            
            # Wait for health check
            for _ in range(30):  # 60 seconds timeout
                if self._check_model_health(model):
                    model.status = ModelStatus.RUNNING
                    return True, f"Model {name} started successfully on port {model.config.port}"
                time.sleep(2)
            
            # Timeout
            self.system_monitor.kill_process(process.pid)
            model.status = ModelStatus.ERROR
            model.pid = None
            return False, "Model failed to start within timeout"
            
        except Exception as e:
            model.status = ModelStatus.ERROR
            model.pid = None
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
                # Check if process still exists
                try:
                    process = psutil.Process(model.pid)
                    if not process.is_running():
                        model.status = ModelStatus.STOPPED
                        model.pid = None
                        continue
                        
                    model.cpu_percent = process.cpu_percent()
                    
                    # Find GPU memory usage
                    for gpu_proc in gpu_processes:
                        if gpu_proc.pid == model.pid:
                            model.gpu_memory_mb = gpu_proc.gpu_memory_mb
                            break
                    
                    # Update uptime
                    if model.start_time:
                        model.uptime_seconds = (datetime.now() - model.start_time).total_seconds()
                    
                    # Health check
                    if not self._check_model_health(model):
                        model.status = ModelStatus.UNHEALTHY
                        
                except psutil.NoSuchProcess:
                    model.status = ModelStatus.STOPPED
                    model.pid = None

    def _check_model_health(self, model: ModelState) -> bool:
        """Check if model is healthy"""
        try:
            import httpx
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"http://localhost:{model.config.port}/health")
                healthy = response.status_code == 200
                model.last_health_check = datetime.now()
                return healthy
        except:
            return False

    def _free_gpu_memory(self, required_mb: float) -> bool:
        """Free GPU memory by killing lower priority processes"""
        processes = self.system_monitor.get_gpu_processes()
        
        # Sort by priority (higher number = lower priority = kill first)
        processes.sort(key=lambda p: p.priority, reverse=True)
        
        for process in processes:
            if process.priority >= 3:  # Only kill medium/low priority
                if self.system_monitor.kill_process(process.pid):
                    time.sleep(2)
                    
                    # Check if we have enough memory now
                    gpus = self.system_monitor.get_gpu_info()
                    if gpus:
                        available = gpus[0].memory_total_mb - gpus[0].memory_used_mb
                        if available >= required_mb:
                            return True
        
        return False

    def aggressive_gpu_cleanup(self, preserve_priority: int = 1) -> Tuple[bool, str]:
        """Aggressively clean up GPU VRAM by killing processes"""
        try:
            initial_gpu_info = self.system_monitor.get_gpu_info()
            if not initial_gpu_info:
                return False, "No GPU found"
            
            initial_used = initial_gpu_info[0].memory_used_mb
            initial_processes = self.system_monitor.get_gpu_processes()
            
            if not initial_processes:
                return True, f"GPU already clean ({initial_used:.0f}MB used)"
            
            killed_count = 0
            freed_memory = 0.0
            
            # Sort processes by priority (kill lowest priority first)
            processes_to_kill = [p for p in initial_processes if p.priority > preserve_priority]
            processes_to_kill.sort(key=lambda p: p.priority, reverse=True)
            
            for process in processes_to_kill:
                try:
                    memory_before = process.gpu_memory_mb
                    if self.system_monitor.kill_process(process.pid):
                        killed_count += 1
                        freed_memory += memory_before
                        time.sleep(1)  # Give process time to die
                        print(f"Killed PID {process.pid} ({process.name}) - freed {memory_before:.0f}MB")
                except Exception as e:
                    print(f"Failed to kill PID {process.pid}: {e}")
            
            # Wait for cleanup to complete
            time.sleep(3)
            
            # Check final state
            final_gpu_info = self.system_monitor.get_gpu_info()
            if final_gpu_info:
                final_used = final_gpu_info[0].memory_used_mb
                actual_freed = initial_used - final_used
                
                return True, (f"Aggressive cleanup complete: "
                            f"killed {killed_count} processes, "
                            f"freed {actual_freed:.0f}MB "
                            f"({initial_used:.0f}MB → {final_used:.0f}MB)")
            else:
                return True, f"Killed {killed_count} processes, freed ~{freed_memory:.0f}MB"
                
        except Exception as e:
            return False, f"Cleanup failed: {str(e)}"

    def force_gpu_cleanup(self) -> Tuple[bool, str]:
        """Force cleanup: Kill ALL GPU processes except priority 1"""
        try:
            processes = self.system_monitor.get_gpu_processes()
            if not processes:
                return True, "No GPU processes found"
            
            # Kill everything except highest priority
            high_priority_processes = [p for p in processes if p.priority == 1]
            processes_to_kill = [p for p in processes if p.priority > 1]
            
            if not processes_to_kill:
                return True, f"Only {len(high_priority_processes)} high-priority processes running"
            
            killed_count = 0
            total_memory = sum(p.gpu_memory_mb for p in processes_to_kill)
            
            print("🚨 FORCE GPU CLEANUP - KILLING ALL NON-CRITICAL PROCESSES")
            
            for process in processes_to_kill:
                try:
                    if self.system_monitor.kill_process(process.pid):
                        killed_count += 1
                        print(f"💥 Killed PID {process.pid} ({process.name}) - {process.gpu_memory_mb:.0f}MB")
                except Exception as e:
                    print(f"Failed to kill PID {process.pid}: {e}")
            
            time.sleep(5)  # Wait for full cleanup
            
            return True, (f"🚨 FORCE CLEANUP: killed {killed_count} processes, "
                        f"freed ~{total_memory:.0f}MB GPU memory")
            
        except Exception as e:
            return False, f"Force cleanup failed: {str(e)}"


# =============================================================================
# TERMINAL UI
# =============================================================================

class TerminalUI:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.running = True
        self.selected_index = 0
        self.current_view = "dashboard"  # dashboard, add_model, help
        self.status_message = ""
        self.status_time = None
        
        # Colors
        self.colors = {}
        
    def init_colors(self):
        """Initialize color pairs"""
        curses.start_color()
        curses.use_default_colors()
        
        # Define color pairs
        curses.init_pair(1, curses.COLOR_GREEN, -1)    # Running
        curses.init_pair(2, curses.COLOR_RED, -1)      # Stopped/Error
        curses.init_pair(3, curses.COLOR_YELLOW, -1)   # Starting/Warning
        curses.init_pair(4, curses.COLOR_BLUE, -1)     # Headers
        curses.init_pair(5, curses.COLOR_CYAN, -1)     # Info
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # Highlight
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Selected
        
        self.colors = {
            'running': curses.color_pair(1) | curses.A_BOLD,
            'stopped': curses.color_pair(2),
            'starting': curses.color_pair(3) | curses.A_BLINK,
            'error': curses.color_pair(2) | curses.A_BOLD,
            'unhealthy': curses.color_pair(3),
            'header': curses.color_pair(4) | curses.A_BOLD,
            'info': curses.color_pair(5),
            'highlight': curses.color_pair(6) | curses.A_BOLD,
            'selected': curses.color_pair(7) | curses.A_BOLD
        }

    def set_status(self, message: str):
        """Set status message"""
        self.status_message = message
        self.status_time = datetime.now()

    def draw_header(self, stdscr):
        """Draw the header"""
        height, width = stdscr.getmaxyx()
        
        # Title
        title = "🚀 VLLM MULTI-MODEL TERMINAL MANAGER 🚀"
        stdscr.addstr(0, (width - len(title)) // 2, title, self.colors['header'])
        
        # GPU Info
        gpus = self.model_manager.system_monitor.get_gpu_info()
        if gpus:
            gpu = gpus[0]
            gpu_info = f"GPU: {gpu.name} | Memory: {gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f}MB ({gpu.memory_used_mb/gpu.memory_total_mb*100:.1f}%) | Util: {gpu.utilization_percent}%"
            if gpu.temperature:
                gpu_info += f" | Temp: {gpu.temperature}°C"
            stdscr.addstr(1, 2, gpu_info, self.colors['info'])
        
        # Separator
        stdscr.addstr(2, 0, "─" * width, self.colors['header'])

    def draw_models_table(self, stdscr, start_y: int):
        """Draw the models table"""
        height, width = stdscr.getmaxyx()
        
        # Table headers
        headers = ["Model", "Status", "Port", "GPU(MB)", "CPU%", "Uptime", "Priority"]
        col_widths = [20, 12, 8, 10, 8, 12, 8]
        
        x = 2
        stdscr.addstr(start_y, x, "MODELS:", self.colors['header'])
        start_y += 1
        
        # Header row
        x = 2
        for i, (header, width_h) in enumerate(zip(headers, col_widths)):
            stdscr.addstr(start_y, x, header.ljust(width_h), self.colors['header'])
            x += width_h + 2
        
        start_y += 1
        stdscr.addstr(start_y, 2, "─" * (sum(col_widths) + len(col_widths) * 2), self.colors['header'])
        start_y += 1
        
        # Model rows
        models = list(self.model_manager.models.values())
        for i, model in enumerate(models):
            if start_y >= height - 3:
                break
                
            # Determine colors based on status
            if model.status == ModelStatus.RUNNING:
                color = self.colors['running']
                status_icon = "🟢"
            elif model.status == ModelStatus.STARTING:
                color = self.colors['starting']
                status_icon = "🟡"
            elif model.status == ModelStatus.ERROR:
                color = self.colors['error']
                status_icon = "🔴"
            elif model.status == ModelStatus.UNHEALTHY:
                color = self.colors['unhealthy']
                status_icon = "🟠"
            else:
                color = self.colors['stopped']
                status_icon = "⚫"
            
            # Highlight selected row
            row_color = self.colors['selected'] if i == self.selected_index else color
            
            # Format uptime
            uptime_str = ""
            if model.uptime_seconds > 0:
                hours = int(model.uptime_seconds // 3600)
                minutes = int((model.uptime_seconds % 3600) // 60)
                uptime_str = f"{hours:02d}:{minutes:02d}"
            
            # Draw row
            x = 2
            values = [
                model.name[:18],
                f"{status_icon} {model.status.value}",
                str(model.config.port),
                f"{model.gpu_memory_mb:.0f}" if model.gpu_memory_mb > 0 else "-",
                f"{model.cpu_percent:.1f}" if model.cpu_percent > 0 else "-",
                uptime_str or "-",
                str(model.config.priority)
            ]
            
            for value, width_v in zip(values, col_widths):
                stdscr.addstr(start_y, x, str(value).ljust(width_v), row_color)
                x += width_v + 2
            
            start_y += 1
        
        return start_y

    def draw_controls(self, stdscr, start_y: int):
        """Draw control instructions"""
        height, width = stdscr.getmaxyx()
        
        controls = [
            "CONTROLS:",
            "↑/↓ - Navigate | ENTER - Start/Stop | a - Add Model | d - Delete Model",
            "r - Refresh | k - Kill Process | c - GPU Cleanup | C - Force Cleanup | h - Help | q - Quit"
        ]
        
        for i, control in enumerate(controls):
            if start_y + i < height - 1:
                color = self.colors['header'] if i == 0 else self.colors['info']
                stdscr.addstr(start_y + i, 2, control, color)

    def draw_status(self, stdscr):
        """Draw status message"""
        height, width = stdscr.getmaxyx()
        
        if self.status_message and self.status_time:
            # Show status for 3 seconds
            if (datetime.now() - self.status_time).total_seconds() < 3:
                stdscr.addstr(height - 1, 2, self.status_message[:width-4], self.colors['highlight'])
            else:
                self.status_message = ""

    def draw_add_model_form(self, stdscr):
        """Draw add model form"""
        height, width = stdscr.getmaxyx()
        
        form_lines = [
            "ADD NEW MODEL:",
            "",
            "Name: ________________",
            "HuggingFace ID: ________________________________",
            "Port: ____",
            "Priority (1-5): _",
            "GPU Memory (0.1-0.9): ___",
            "Max Model Length: ____",
            "",
            "Press ENTER to add, ESC to cancel"
        ]
        
        start_y = (height - len(form_lines)) // 2
        for i, line in enumerate(form_lines):
            color = self.colors['header'] if i == 0 else self.colors['info']
            stdscr.addstr(start_y + i, (width - len(line)) // 2, line, color)

    def draw_help_screen(self, stdscr):
        """Draw comprehensive help screen"""
        height, width = stdscr.getmaxyx()
        
        help_lines = [
            "🚀 VLLM MULTI-MODEL TERMINAL MANAGER - HELP 🚀",
            "",
            "NAVIGATION:",
            "  ↑/↓ Arrow Keys    - Navigate between models",
            "  ENTER or SPACE    - Start/Stop selected model",
            "  ESC               - Exit current screen/form",
            "",
            "MODEL MANAGEMENT:",
            "  a                 - Add new model configuration",
            "  d                 - Delete selected model (stops if running)",
            "  k                 - Kill selected model process immediately",
            "  r                 - Refresh all model statuses",
            "",
            "GPU MEMORY CLEANUP:",
            "  c                 - Clean GPU memory (kills priority 2-5 processes)",
            "  C (Shift+C)       - FORCE cleanup (kills ALL except priority 1)",
            "",
            "PRIORITY SYSTEM:",
            "  Priority 1        - Critical (never killed by cleanup)",
            "  Priority 2        - High importance",
            "  Priority 3        - Normal (default)",
            "  Priority 4        - Low importance", 
            "  Priority 5        - Disposable (killed first)",
            "",
            "COMMAND LINE INTERFACE:",
            "  ./vllm add <name> <model_id> --port <port>",
            "  ./vllm start <name>",
            "  ./vllm stop <name>",
            "  ./vllm list",
            "  ./vllm status",
            "  ./vllm cleanup",
            "  ./vllm force",
            "",
            "EXAMPLES:",
            "  ./vllm add llama-3.1-8b meta-llama/Llama-3.1-8B-Instruct --port 9798",
            "  ./vllm start llama-3.1-8b",
            "",
            "Press any key to return to dashboard..."
        ]
        
        # Calculate starting position to center vertically
        start_y = max(1, (height - len(help_lines)) // 2)
        
        for i, line in enumerate(help_lines):
            if start_y + i >= height - 1:
                break
                
            # Color coding
            if line.startswith("🚀") or line.endswith("🚀"):
                color = self.colors['header']
            elif line.endswith(":") and not line.startswith("  "):
                color = self.colors['highlight']
            elif line.startswith("  ./vllm"):
                color = self.colors['error']  # Use error color for command examples
            elif line.startswith("  "):
                color = self.colors['info']
            else:
                color = self.colors['normal']
            
            # Center the line
            x_pos = max(2, (width - len(line)) // 2)
            stdscr.addstr(start_y + i, x_pos, line, color)

    def run(self, stdscr):
        """Main UI loop"""
        # Initialize
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        stdscr.timeout(100)  # 100ms timeout
        
        self.init_colors()
        
        # Start stats update thread
        def update_stats():
            while self.running:
                self.model_manager.update_model_stats()
                time.sleep(2)
        
        stats_thread = threading.Thread(target=update_stats, daemon=True)
        stats_thread.start()
        
        last_refresh = 0
        
        while self.running:
            current_time = time.time()
            
            # Refresh screen every 1 second or on input
            key = stdscr.getch()
            if key != -1 or current_time - last_refresh > 1:
                stdscr.clear()
                
                if self.current_view == "dashboard":
                    self.draw_header(stdscr)
                    start_y = self.draw_models_table(stdscr, 4)
                    self.draw_controls(stdscr, start_y + 2)
                    self.draw_status(stdscr)
                elif self.current_view == "add_model":
                    self.draw_add_model_form(stdscr)
                elif self.current_view == "help":
                    self.draw_help_screen(stdscr)
                
                stdscr.refresh()
                last_refresh = current_time
            
            # Handle input
            if key != -1:
                self.handle_input(key)
            
            time.sleep(0.05)

    def handle_input(self, key):
        """Handle keyboard input"""
        if self.current_view == "dashboard":
            models = list(self.model_manager.models.values())
            
            if key == ord('q'):
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
            elif key == ord('a'):
                self.current_view = "add_model"
            elif key == ord('d'):
                if models and 0 <= self.selected_index < len(models):
                    model = models[self.selected_index]
                    if self.model_manager.remove_model(model.name):
                        self.set_status(f"Model {model.name} removed")
                        self.selected_index = max(0, self.selected_index - 1)
            elif key == ord('r'):
                self.set_status("Refreshed")
            elif key == ord('k'):
                # Kill selected model process
                if models and 0 <= self.selected_index < len(models):
                    model = models[self.selected_index]
                    if model.pid:
                        if self.model_manager.system_monitor.kill_process(model.pid):
                            self.set_status(f"Killed process {model.pid}")
                        else:
                            self.set_status("Failed to kill process")
            elif key == ord('c'):
                # Aggressive GPU cleanup
                success, msg = self.model_manager.aggressive_gpu_cleanup(preserve_priority=1)
                self.set_status(f"🧹 {msg}")
            elif key == ord('C'):
                # Force GPU cleanup (Shift+C)
                success, msg = self.model_manager.force_gpu_cleanup()
                self.set_status(f"💥 {msg}")
            elif key == ord('h') or key == ord('H'):
                # Show help screen
                self.current_view = "help"
        
        elif self.current_view == "add_model":
            if key == 27:  # ESC
                self.current_view = "dashboard"
        
        elif self.current_view == "help":
            # Any key returns to dashboard from help
            self.current_view = "dashboard"


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize
    model_manager = ModelManager()
    ui = TerminalUI(model_manager)
    
    try:
        curses.wrapper(ui.run)
    except KeyboardInterrupt:
        pass
    
    print("👋 VLLM Manager terminated")


if __name__ == "__main__":
    main()