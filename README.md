# ollama-dl v2

An advanced CLI for managing Ollama models — download, inspect, verify, archive, and batch-install.

---

## Build

```bash
go mod tidy
go build -o ollama-dl .
```

Cross-compile (no CGO needed):

```bash
GOOS=linux  GOARCH=amd64  go build -o ollama-dl-linux-amd64  .
GOOS=linux  GOARCH=arm64  go build -o ollama-dl-linux-arm64  .
GOOS=darwin GOARCH=arm64  go build -o ollama-dl-darwin-arm64 .
GOOS=windows GOARCH=amd64 go build -o ollama-dl.exe          .
```

---

## Commands

### `urls` — Direct download links

```bash
ollama-dl urls llama3:8b
ollama-dl urls --json deepseek-coder-v2:latest
ollama-dl urls --registry https://my-registry.local mymodel:v1
```

Prints HTTPS URLs for every blob, the manifest JSON, and ready-to-use `wget -c` commands.

---

### `pull` — Built-in parallel downloader with progress bars

```bash
ollama-dl pull llama3:8b
ollama-dl pull -o ./downloads -j 5 deepseek-coder-v2:latest
```

- Resumable downloads (sends `Range` header, continues from partial `.part` files)
- Live per-blob progress bars with ETA and speed (via `mpb`)
- Configurable concurrency and retry count
- Saves `manifest.json` alongside the blobs automatically

Flags:
- `-o, --out <dir>` — output directory (default: `<repo>-<tag>`)
- `-j, --concurrency <n>` — parallel downloads (default: 3)
- `--retries <n>` — max retries per blob (default: 3)
- `--registry <url>` — override registry base URL

---

### `install` — Install into local Ollama store

```bash
ollama-dl install ./llama3-8b my-llama:local
ollama-dl install ./model --symlink my-llama:local   # symlink blobs (saves disk)
ollama-dl install ./model --ollama-dir /custom/path my-llama:local
```

Copies (or symlinks) all blobs to `~/.ollama/models/blobs/` and registers the manifest. The model is immediately available to `ollama run` — no restart needed.

---

### `info` — Model architecture & metadata

```bash
ollama-dl info llama3:8b
ollama-dl info deepseek-coder-v2:16b-lite-instruct-q4_K_M
```

Downloads and parses the config blob, displaying:

```
  Model Family          llama
  Architecture          llama
  Parameters            8B
  Format                gguf
  Quantization          Q4_0
  Context Length        8192 tokens
  Stop Tokens           <|start_header_id|> <|end_header_id|> <|eot_id|>

  Blobs:                5
  Total size:           4.7 GB

  Layers:
    model               4.7 GB
    template            1.4 KB
    params              96 B
```

---

### `diff` — Compare two model versions

```bash
ollama-dl diff llama3:8b llama3:8b-instruct-q4_K_M
ollama-dl diff mistral:7b mistral:7b-instruct-v0.2
```

Shows exactly which blobs changed between two versions, how much data you'd need to re-download, and how much is shared (already on disk).

```
  + model               4.4 GB    sha256:3c37...
  - model               4.7 GB    sha256:6a0a...

  Shared blobs    : 3 (no download needed)
  Added in B      : 1  (+4.4 GB)
  Removed from A  : 1  (-4.7 GB)
  Net change: -307.2 MB
```

---

### `verify` — SHA256 integrity check

```bash
ollama-dl verify                       # all installed models
ollama-dl verify llama3:8b             # single model
ollama-dl verify --fix llama3:8b       # remove corrupt blobs
```

Reads every installed model, SHA256-hashes each blob on disk, and compares it against the digest in the manifest. Use `--fix` to delete corrupt blobs so Ollama can re-pull them cleanly.

---

### `search` — Browse the Ollama library

```bash
ollama-dl search llama
ollama-dl search "code generation"
ollama-dl search --json mistral
```

Queries the Ollama library API and lists matching models with their tags and pull counts. Use the result names directly with `pull` or `urls`.

---

### `pack` — Create a portable archive

```bash
ollama-dl pack llama3:8b
ollama-dl pack llama3:8b -o llama3-8b-backup.tar.zst
ollama-dl pack llama3:8b --compression 9    # slower but smaller
```

Bundles the model's blobs + manifest into a single `.tar.zst` file. Ideal for:
- Air-gapped machine transfers (copy one file over USB/SCP)
- Long-term model archival
- Sharing specific quantizations with teammates

The archive is self-contained — no network access needed to install from it.

---

### `unpack` — Install from an archive

```bash
ollama-dl unpack llama3-8b-20240115.tar.zst --name llama3:local
ollama-dl unpack archive.tar.zst --inspect      # list contents only
```

Extracts the archive and installs it into the Ollama store, exactly like `install`.

---

### `batch` — Download multiple models at once

```bash
ollama-dl batch models.yaml
ollama-dl batch models.yaml --dry-run       # preview without downloading
ollama-dl batch models.yaml -j 5            # 5 parallel downloads per model
```

Reads a YAML (or JSON) manifest:

```yaml
out_dir: ./downloaded-models

models:
  - model: llama3:8b
    auto_install: true

  - model: mistral:7b-instruct-v0.2-q5_K_M
    name: mistral-q5:local        # install under this name
    auto_install: true

  - model: deepseek-coder-v2:latest
    auto_install: false            # download only, install later

  # Private registry support
  - model: myorg/custom-model:v2
    registry: https://my-registry.example.com
    auto_install: true
```

Blobs shared across models are downloaded only once and symlinked for the rest (deduplication).

---

## Environment Variables

| Variable          | Effect                                  |
|-------------------|-----------------------------------------|
| `OLLAMA_MODELS`   | Override Ollama models directory        |

---

## Typical Workflows

### Air-gapped transfer

```bash
# Machine with internet:
ollama-dl pull llama3:8b -o ./llama3
ollama-dl pack llama3:8b -o llama3-8b.tar.zst

# Transfer llama3-8b.tar.zst to air-gapped machine, then:
ollama-dl unpack llama3-8b.tar.zst --name llama3:8b
ollama run llama3:8b
```

### Verify and repair a suspect install

```bash
ollama-dl verify --fix llama3:8b
ollama pull llama3:8b    # re-pull only the corrupt blobs
```

### Audit total disk usage

```bash
ollama-dl verify          # lists all models + blob sizes
```

### Compare before upgrading

```bash
ollama-dl diff llama3:8b llama3:8b-instruct-fp16
# Decide if the extra download is worth it
```
