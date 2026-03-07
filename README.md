```
 ██╗      ██████╗  ██████╗ █████╗ ██╗          ██████╗██╗     ██╗
 ██║     ██╔═══██╗██╔════╝██╔══██╗██║         ██╔════╝██║     ██║
 ██║     ██║   ██║██║     ███████║██║  █████╗ ██║     ██║     ██║
 ██║     ██║   ██║██║     ██╔══██║██║  ╚════╝ ██║     ██║     ██║
 ███████╗╚██████╔╝╚██████╗██║  ██║███████╗    ╚██████╗███████╗██║
 ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝     ╚═════╝╚══════╝╚═╝
```

**Local-first AI coding agent powered by Ollama.**

Zero external dependencies. Runs entirely on your machine.

[English](README.md) | [日本語](README.ja.md) | [やさしいにほんご](README.easy-ja.md)

---

## What is this?

Local CLI is an AI coding agent that runs locally using [Ollama](https://ollama.com). It can read, write, edit files, run shell commands, search code, and fetch web pages — all through natural language.

Think of it as a local, offline-capable alternative to cloud-based AI coding assistants.

### Key Features

- **Agent loop** — LLM autonomously calls tools (read, write, edit, bash, glob, grep) to complete tasks
- **Streaming responses** — Real-time token-by-token output
- **Multi-provider** — Ollama (local) and Claude API support with runtime switching
- **Model management** — Install, delete, and switch models from the CLI or desktop app
- **RAG engine** — Index your codebase for context-aware responses
- **Git checkpoints** — Create and rollback to safe points during edits
- **Session persistence** — Save and resume conversations
- **Desktop GUI** — Electron app with terminal-style UI
- **Zero dependencies** — Python stdlib only (no pip install needed for the core)

## Download

### Desktop App (pre-built)

Download the latest release for your platform from **[GitHub Releases](https://github.com/lutelute/local-cli/releases)**:

| Platform | File |
|----------|------|
| macOS | `Local-CLI-x.x.x-universal.dmg` |
| Windows | `Local-CLI-Setup-x.x.x.exe` |
| Linux | `Local-CLI-x.x.x.AppImage` or `.deb` |

> **Note:** [Ollama](https://ollama.com) must be installed and running on your machine. The desktop app connects to Ollama locally.

#### macOS: "App is damaged" or "unidentified developer" warning

The app is not signed with an Apple Developer certificate. macOS Gatekeeper will block it on first launch. To allow it:

```bash
xattr -cr /Applications/Local\ CLI.app
```

Or: **System Settings > Privacy & Security > scroll down > click "Open Anyway"**.

### CLI (from source)

```bash
# Requirements: Python 3.10+, Ollama, Git

# 1. Clone
git clone https://github.com/lutelute/local-cli.git
cd local-cli

# 2. Run directly
python -m local_cli

# Or install as a command
pip install -e .
local-cli
```

### Interactive Model Selection

```bash
# Pick a model at startup
local-cli --select-model

# Use a specific model
local-cli --model qwen3:8b
```

## Update

Updates are checked automatically on startup. When an update is available, you'll see a notification.

```bash
# CLI: update from terminal
local-cli --update

# CLI: update from within the REPL
/update

# Desktop: click "Install update" in the notification bar
```

Since `pip install -e .` links to the source directory, `git pull` is all that's needed. The `/update` command does this automatically.

## Architecture

```
local-cli/
├── local_cli/
│   ├── __main__.py          # Entry point
│   ├── agent.py             # Agent loop (LLM <-> tools)
│   ├── cli.py               # REPL + slash commands
│   ├── config.py            # Configuration (CLI > env > file > defaults)
│   ├── server.py            # JSON-line server for desktop GUI
│   ├── ollama_client.py     # Ollama REST API client
│   ├── orchestrator.py      # Multi-provider orchestration
│   ├── model_catalog.py     # Curated model catalog + cache
│   ├── model_search.py      # Live search from ollama.com
│   ├── model_manager.py     # Install/delete/info operations
│   ├── model_registry.py    # Task-to-model routing
│   ├── model_selector.py    # Interactive TUI model picker
│   ├── rag.py               # Retrieval-augmented generation
│   ├── git_ops.py           # Git checkpoint/rollback
│   ├── session.py           # Session save/load
│   ├── security.py          # Input validation
│   ├── updater.py           # Self-update (git pull)
│   ├── providers/           # LLM provider abstraction
│   │   ├── base.py          # Abstract LLMProvider
│   │   ├── ollama_provider.py
│   │   ├── claude_provider.py
│   │   ├── message_converter.py
│   │   └── sse_parser.py
│   └── tools/               # Agent tools
│       ├── bash_tool.py     # Shell command execution
│       ├── read_tool.py     # File reading
│       ├── write_tool.py    # File creation
│       ├── edit_tool.py     # String replacement editing
│       ├── glob_tool.py     # File pattern search
│       ├── grep_tool.py     # Content search
│       ├── web_fetch_tool.py
│       └── ask_user_tool.py
├── desktop/                 # Electron + React + Vite
│   ├── electron/            # Main process + preload
│   ├── src/                 # React UI components
│   └── build/               # App icons
└── tests/                   # 961 tests
```

## Tools

The agent has access to these tools:

| Tool | Description |
|------|-------------|
| `bash` | Run shell commands |
| `read` | Read file contents |
| `write` | Create or overwrite files |
| `edit` | Find-and-replace editing |
| `glob` | Find files by pattern |
| `grep` | Search file contents |
| `web_fetch` | Fetch web pages |
| `ask_user` | Ask the user a question |

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/model <name>` | Switch model |
| `/models` | Open interactive model selector |
| `/status` | Show connection and model info |
| `/provider [name]` | Switch or show LLM provider |
| `/brain [model]` | Set orchestrator brain model |
| `/install <model>` | Download a model |
| `/uninstall <model>` | Delete a model |
| `/info <model>` | Show model details |
| `/running` | List models loaded in VRAM |
| `/checkpoint` | Create git checkpoint |
| `/rollback [tag]` | Rollback to checkpoint |
| `/save` | Save session |
| `/update` | Check for updates and pull latest version |
| `/clear` | Clear conversation |
| `/exit` | Quit |

## Desktop App

The desktop app provides a terminal-style GUI with:

- Streaming chat with tool call display
- Model picker with **Catalog** (curated models by category) and **Discover** (live search from ollama.com)
- Download, switch, and delete models
- Auto-update notification bar
- ASCII art welcome banner

Communication between Electron and Python uses stdin/stdout JSON lines — no network server, no API dependencies.

### Run from Source

```bash
cd desktop
npm install
npm run dev          # Development mode (hot reload)
# or
npx vite build && npx electron .   # Production preview
```

### Build Installers

```bash
cd desktop
npm run build        # Build for current platform
npm run build:mac    # macOS → .dmg + .zip (universal)
npm run build:win    # Windows → NSIS installer
npm run build:linux  # Linux → AppImage + .deb
```

Output goes to `desktop/dist/`. The installer bundles the Python source and Electron runtime.

### Release to GitHub

```bash
cd desktop

# Build and publish to GitHub Releases (requires GH_TOKEN)
export GH_TOKEN=your_github_token
npx electron-builder --publish always
```

This uploads the installer to a draft release on GitHub. Edit and publish the release from the GitHub UI.

## Configuration

Configuration is resolved in order: CLI flags > environment variables > config file > defaults.

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--model` | `LOCAL_CLI_MODEL` | `qwen3:8b` | Model to use |
| `--provider` | `LOCAL_CLI_PROVIDER` | `ollama` | LLM provider |
| `--debug` | `LOCAL_CLI_DEBUG` | `false` | Debug output |
| `--rag` | — | `false` | Enable RAG |
| `--rag-path` | — | `.` | Directory to index |
| `--select-model` | — | `false` | Interactive model picker |
| `--server` | — | `false` | JSON-line server mode |
| `--update` | — | `false` | Check for and install updates |

## Recommended Models

| Model | Size | Best For |
|-------|------|----------|
| `qwen3:8b` | 5.2 GB | General use, tool calling |
| `qwen2.5-coder:7b` | 4.7 GB | Code generation |
| `qwen3:30b` | 18.5 GB | Complex reasoning, agents |
| `deepseek-r1:14b` | 9.0 GB | Chain-of-thought reasoning |
| `qwen3:0.6b` | 0.5 GB | Quick testing |

## Tests

```bash
python -m pytest tests/ -q
# 961 passed
```

## License

MIT
