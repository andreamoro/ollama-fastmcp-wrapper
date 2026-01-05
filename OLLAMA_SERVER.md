# Ollama Server Configuration

This document covers server-side configuration for Ollama. These settings must be configured on the machine running `ollama serve` and cannot be controlled by the wrapper.

## Environment Variables

Set these **before** starting `ollama serve`. They cannot be changed at runtime.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_NUM_PARALLEL` | 1 | Concurrent requests per model. Increase for parallel inference (requires more VRAM). |
| `OLLAMA_MAX_LOADED_MODELS` | 1 | Models kept in VRAM simultaneously. Useful for multi-model workflows. |
| `OLLAMA_MAX_QUEUE` | 512 | Maximum queued requests before rejection. |
| `OLLAMA_HOST` | `127.0.0.1:11434` | Address Ollama listens on. Use `0.0.0.0:11434` for network access. |
| `OLLAMA_ORIGINS` | (none) | Allowed CORS origins (comma-separated). |
| `OLLAMA_MODELS` | `~/.ollama/models` | Directory for model storage. |

## Examples

### Basic Setup

```bash
# Start with defaults
ollama serve
```

### Enable Parallel Inference

```bash
# Allow 2 concurrent requests (requires ~2x VRAM per model)
export OLLAMA_NUM_PARALLEL=2
ollama serve
```

### Multi-Model Setup

```bash
# Keep 2 models loaded for faster switching
export OLLAMA_MAX_LOADED_MODELS=2
ollama serve
```

### Network Access

```bash
# Allow connections from other machines
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

### Production Setup

```bash
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_MAX_QUEUE=1024
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

## Systemd Service

For persistent configuration on Linux servers:

```bash
sudo systemctl edit ollama.service
```

Add environment variables:

```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## Remote Ollama via SSH Tunnel

When running Ollama on a remote server, use SSH port forwarding:

```bash
# On local machine: forward local:11435 to remote:11434
ssh -L 11435:localhost:11434 user@remote-server

# Then connect the wrapper to the tunnel
python ollama_wrapper.py api --ollama-port 11435 --ollama-label "remote-server"
```

For VS Code Remote, use the Ports tab to forward port 11434 from the remote to a local port.

## VRAM Considerations

| Setting | VRAM Impact |
|---------|-------------|
| `OLLAMA_NUM_PARALLEL=2` | ~2x per model |
| `OLLAMA_NUM_PARALLEL=4` | ~4x per model |
| `OLLAMA_MAX_LOADED_MODELS=2` | 2 models in memory |

Example: A 7B model (~4GB VRAM) with `NUM_PARALLEL=2` needs ~8GB VRAM.

## Troubleshooting

**Requests are queued (slow parallel performance):**
- Check `OLLAMA_NUM_PARALLEL` is set before `ollama serve`
- Verify with: `echo $OLLAMA_NUM_PARALLEL`

**Out of memory errors:**
- Reduce `OLLAMA_NUM_PARALLEL`
- Reduce `OLLAMA_MAX_LOADED_MODELS`
- Use smaller models (e.g., 3B instead of 7B)

**Connection refused from network:**
- Set `OLLAMA_HOST=0.0.0.0:11434`
- Check firewall allows port 11434
