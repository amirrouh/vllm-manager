# HuggingFace Authentication Guide for VLLM Manager

## Problem Summary
When starting models that require HuggingFace authentication (like Qwen2.5), the VLLM Manager may fail with:
```
huggingface_hub.errors.HfHubHTTPError: 401 Client Error: Unauthorized
Invalid credentials in Authorization header
```

## Root Cause
The VLLM Manager was not properly inheriting HuggingFace authentication tokens from the environment, causing authentication failures when downloading gated models.

## Solution Implemented

### 1. Token Authentication Fix
The manager now reads the HuggingFace token directly from the cache file (`~/.cache/huggingface/token`) instead of relying solely on environment variables.

**Key Changes:**
- Reads token from `~/.cache/huggingface/token` (primary source)
- Falls back to `HF_TOKEN` environment variable if cache is unavailable
- Properly passes token to subprocess environment

### 2. GPU Memory Optimization
Qwen2.5 models require more GPU memory for KV cache. The manager now:
- Automatically increases GPU memory utilization to 80% for Qwen2.5 models
- Updates vLLM command parameters dynamically

## Setup Instructions

### For New Users

1. **Login to HuggingFace:**
   ```bash
   huggingface-cli login --token YOUR_TOKEN_HERE
   ```

2. **Add Your Model:**
   ```bash
   ./vm add qwen2.5-7b Qwen/Qwen2.5-7B-Instruct --port 8001
   ```

3. **Start the Model:**
   ```bash
   ./vm start qwen2.5-7b
   ```

### For Existing Users

1. **Ensure Valid HuggingFace Login:**
   ```bash
   # Check current login status
   huggingface-cli whoami

   # If not logged in or login is invalid:
   huggingface-cli logout
   huggingface-cli login --token YOUR_VALID_TOKEN
   ```

2. **Clear Old Environment Variables (Optional):**
   ```bash
   # Remove conflicting environment variables
   unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
   ```

## Troubleshooting

### Authentication Issues

**Symptoms:**
- 401 Unauthorized errors
- "Invalid credentials in Authorization header"

**Solutions:**
1. Verify HuggingFace login:
   ```bash
   huggingface-cli whoami
   ```

2. Re-login with correct token:
   ```bash
   huggingface-cli logout
   huggingface-cli login --token YOUR_TOKEN
   ```

3. Check token permissions:
   - Ensure your HuggingFace account has access to the model
   - Some models require specific access requests

### GPU Memory Issues

**Symptoms:**
- "No available memory for the cache blocks"
- Process starts but fails during initialization

**Solutions:**
1. The manager now automatically handles this for Qwen2.5 models
2. For other models, you can manually increase GPU memory:
   ```bash
   # Edit model configuration in models_config.json
   "gpu_memory_utilization": 0.8
   ```

### Model-Specific Notes

#### Qwen2.5 Models
- **Required GPU Memory:** Higher than average (80% utilization recommended)
- **Authentication:** Required (gated model)
- **Trust Remote Code:** Required for some variants

#### Other Gated Models
- Follow the same authentication setup
- GPU memory requirements vary by model size

## Best Practices

1. **Always use `huggingface-cli login`** rather than manual environment variables
2. **Verify model access** on HuggingFace before attempting to download
3. **Monitor GPU memory** usage, especially for larger models
4. **Use the manager's built-in features** rather than manual vLLM commands

## Testing Authentication

To test if your authentication is working:

```bash
# Test login status
huggingface-cli whoami

# Test token validity
python -c "
import requests
import os
with open(os.path.expanduser('~/.cache/huggingface/token'), 'r') as f:
    token = f.read().strip()
headers = {'Authorization': f'Bearer {token}'}
response = requests.get('https://huggingface.co/api/whoami', headers=headers)
print('Status:', response.status_code)
print('Response:', response.text)
"
```

## Future Considerations

- The fix is automatic and requires no user intervention
- Token reading from cache is more reliable than environment variables
- GPU memory optimization is handled automatically for known problematic models
- The solution is backward compatible with existing configurations