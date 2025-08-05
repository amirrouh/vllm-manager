#!/usr/bin/env python3
"""
VLLM CLI - Command Line Interface for Model Management
Clean and organized CLI commands
"""

import argparse
import json
import sys
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from vllm_terminal_manager import ModelManager, ModelConfig


def add_model_command(args):
    """Add a new model"""
    manager = ModelManager()
    
    config = ModelConfig(
        name=args.name,
        huggingface_id=args.model_id,
        port=args.port,
        priority=args.priority,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_len,
        tensor_parallel_size=args.tensor_parallel
    )
    
    if manager.add_model(config):
        print(f"✅ Model '{args.name}' added successfully")
        print(f"   HuggingFace ID: {args.model_id}")
        print(f"   Port: {args.port}")
        print(f"   Priority: {args.priority}")
        print(f"   GPU Memory: {args.gpu_memory}")
    else:
        print(f"❌ Model '{args.name}' already exists")
        return 1
    
    return 0


def list_models_command(args):
    """List all models"""
    manager = ModelManager()
    
    if not manager.models:
        print("No models configured")
        return 0
    
    print("Configured Models:")
    print("─" * 80)
    
    for model in manager.models.values():
        status_icon = {
            "running": "🟢",
            "starting": "🟡", 
            "stopped": "⚫",
            "error": "🔴",
            "unhealthy": "🟠"
        }.get(model.status.value, "⚫")
        
        print(f"{status_icon} {model.name}")
        print(f"   Model: {model.config.huggingface_id}")
        print(f"   Port: {model.config.port}")
        print(f"   Priority: {model.config.priority}")
        print(f"   Status: {model.status.value}")
        if model.pid:
            print(f"   PID: {model.pid}")
        print()
    
    return 0


def start_model_command(args):
    """Start a model"""
    manager = ModelManager()
    
    success, message = manager.start_model(args.name)
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        return 1
    
    return 0


def stop_model_command(args):
    """Stop a model"""
    manager = ModelManager()
    
    success, message = manager.stop_model(args.name)
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        return 1
    
    return 0


def remove_model_command(args):
    """Remove a model"""
    manager = ModelManager()
    
    if manager.remove_model(args.name):
        print(f"✅ Model '{args.name}' removed")
    else:
        print(f"❌ Model '{args.name}' not found")
        return 1
    
    return 0


def status_command(args):
    """Show system status"""
    manager = ModelManager()
    manager.update_model_stats()
    
    # GPU Info
    gpus = manager.system_monitor.get_gpu_info()
    if gpus:
        gpu = gpus[0]
        print("GPU Status:")
        print(f"  Name: {gpu.name}")
        print(f"  Memory: {gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f}MB ({gpu.memory_used_mb/gpu.memory_total_mb*100:.1f}%)")
        print(f"  Utilization: {gpu.utilization_percent}%")
        if gpu.temperature:
            print(f"  Temperature: {gpu.temperature}°C")
        print()
    
    # Running Models
    running_models = [m for m in manager.models.values() if m.status.value in ["running", "starting"]]
    if running_models:
        print("Running Models:")
        for model in running_models:
            print(f"  🟢 {model.name} (port {model.config.port}, PID {model.pid})")
            print(f"     GPU Memory: {model.gpu_memory_mb:.0f}MB")
            print(f"     CPU: {model.cpu_percent:.1f}%")
            if model.uptime_seconds:
                hours = int(model.uptime_seconds // 3600)
                minutes = int((model.uptime_seconds % 3600) // 60)
                print(f"     Uptime: {hours:02d}:{minutes:02d}")
        print()
    
    # GPU Processes
    processes = manager.system_monitor.get_gpu_processes()
    if processes:
        print("GPU Processes:")
        for proc in processes:
            print(f"  PID {proc.pid}: {proc.name}")
            print(f"    GPU Memory: {proc.gpu_memory_mb:.0f}MB")
            print(f"    CPU: {proc.cpu_percent:.1f}%")
        print()
    
    return 0


def cleanup_command(args):
    """Aggressive GPU cleanup command"""
    manager = ModelManager()
    
    print("🧹 Starting aggressive GPU cleanup...")
    success, message = manager.aggressive_gpu_cleanup(preserve_priority=args.preserve_priority)
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        return 1
    
    return 0


def force_command(args):
    """Force GPU cleanup command"""
    manager = ModelManager()
    
    print("💥 Starting FORCE GPU cleanup...")
    print("⚠️  This will kill ALL non-critical GPU processes!")
    
    try:
        response = input("Are you sure? (type 'FORCE' to confirm): ")
        if response != "FORCE":
            print("Aborted.")
            return 0
    except KeyboardInterrupt:
        print("\nAborted.")
        return 0
    
    success, message = manager.force_gpu_cleanup()
    
    if success:
        print(f"💥 {message}")
    else:
        print(f"❌ {message}")
        return 1
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="VLLM Multi-Model Manager CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s add mistral-7b mistralai/Mistral-7B-Instruct-v0.2 --port 8001 --priority 2
  %(prog)s start mistral-7b  
  %(prog)s list
  %(prog)s status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Add model command
    add_parser = subparsers.add_parser('add', 
        help='Add a new model configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./vllm add llama-3.1-8b meta-llama/Llama-3.1-8B-Instruct --port 9798
  ./vllm add mistral-7b mistralai/Mistral-7B-Instruct-v0.2 --port 8001 --priority 2
  ./vllm add codellama codellama/CodeLlama-7b-Instruct-hf --port 8002 --gpu-memory 0.5 --max-len 4096
        """)
    add_parser.add_argument('name', help='Short name for the model (e.g., "llama-3.1-8b")')
    add_parser.add_argument('model_id', help='HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")')
    add_parser.add_argument('--port', type=int, required=True, help='Port number for the model server')
    add_parser.add_argument('--priority', type=int, default=3, choices=[1,2,3,4,5], 
                           help='Process priority: 1=critical (never killed), 2=high, 3=normal, 4=low, 5=disposable')
    add_parser.add_argument('--gpu-memory', type=float, default=0.3, 
                           help='GPU memory utilization ratio (0.1-0.9, default: 0.3)')
    add_parser.add_argument('--max-len', type=int, default=2048, 
                           help='Maximum sequence length (default: 2048)')
    add_parser.add_argument('--tensor-parallel', type=int, default=1,
                           help='Number of GPUs for tensor parallelism (default: 1)')
    add_parser.set_defaults(func=add_model_command)
    
    # List models command
    list_parser = subparsers.add_parser('list', 
        help='List all configured models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./vllm list                    # Show all models with their status
        """)
    list_parser.set_defaults(func=list_models_command)
    
    # Start model command
    start_parser = subparsers.add_parser('start', 
        help='Start a configured model server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./vllm start llama-3.1-8b      # Start the llama-3.1-8b model
  ./vllm start mistral-7b        # Start the mistral-7b model
        """)
    start_parser.add_argument('name', help='Model name to start (use "list" command to see available models)')
    start_parser.set_defaults(func=start_model_command)
    
    # Stop model command
    stop_parser = subparsers.add_parser('stop', 
        help='Stop a running model server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./vllm stop llama-3.1-8b       # Stop the llama-3.1-8b model
  ./vllm stop mistral-7b         # Stop the mistral-7b model
        """)
    stop_parser.add_argument('name', help='Model name to stop (must be currently running)')
    stop_parser.set_defaults(func=stop_model_command)
    
    # Remove model command
    remove_parser = subparsers.add_parser('remove', 
        help='Remove a model configuration (stops it first if running)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./vllm remove llama-3.1-8b     # Remove llama-3.1-8b configuration
  ./vllm remove mistral-7b       # Remove mistral-7b configuration
        """)
    remove_parser.add_argument('name', help='Model name to remove from configuration')
    remove_parser.set_defaults(func=remove_model_command)
    
    # Status command
    status_parser = subparsers.add_parser('status', 
        help='Show system and model status',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./vllm status                  # Show GPU usage, memory, and model statuses
        """)
    status_parser.set_defaults(func=status_command)
    
    # GPU cleanup commands
    cleanup_parser = subparsers.add_parser('cleanup', 
        help='Clean up GPU memory by killing low-priority processes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./vllm cleanup                 # Kill processes with priority > 1 (preserves critical)
  ./vllm cleanup --preserve-priority 2  # Kill processes with priority > 2
        """)
    cleanup_parser.add_argument('--preserve-priority', type=int, default=1,
                               help='Preserve processes with this priority and higher (1=critical, 5=disposable)')
    cleanup_parser.set_defaults(func=cleanup_command)
    
    force_parser = subparsers.add_parser('force', 
        help='Force GPU cleanup - kill ALL non-critical processes (DANGEROUS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./vllm force                   # Kill ALL processes except priority 1 (requires typing 'FORCE')
  
WARNING: This command kills ALL GPU processes except priority 1 (critical).
You must type 'FORCE' to confirm this destructive operation.
        """)
    force_parser.set_defaults(func=force_command)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())