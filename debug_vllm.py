#!/usr/bin/env python3
"""Debug script to test vLLM model loading directly"""

import os
import sys
import subprocess

# Ensure HF_TOKEN is set
if 'HF_TOKEN' in os.environ:
    print(f"✅ HF_TOKEN is set: {os.environ['HF_TOKEN'][:10]}...")
else:
    print("❌ HF_TOKEN not found in environment")

model = "mistralai/Mistral-7B-Instruct-v0.2"
port = "8001"

print(f"\n🔍 Testing vLLM with model: {model}")
print(f"📍 Port: {port}")
print("-" * 60)

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", model,
    "--host", "0.0.0.0",
    "--port", port,
    "--trust-remote-code",
    "--gpu-memory-utilization", "0.9",
    "--max-model-len", "1024",
    "--tensor-parallel-size", "1",
    "--enforce-eager",
    "--disable-log-requests"
]

print(f"\n📋 Command: {' '.join(cmd)}")
print("\n🚀 Starting vLLM (this may take a few minutes)...")
print("-" * 60)

try:
    # Run with full output to see errors
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
        if process.poll() is not None:
            break
    
    returncode = process.wait()
    if returncode != 0:
        print(f"\n❌ Process exited with code: {returncode}")
    
except KeyboardInterrupt:
    print("\n\n⚠️ Interrupted by user")
    process.terminate()
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)