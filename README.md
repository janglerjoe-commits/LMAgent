# LMAgent


[![Watch the demo](https://img.youtube.com/vi/GNQP6M53c5A/hqdefault.jpg)](https://www.youtube.com/watch?v=GNQP6M53c5A)

A cross-platform, locally-hosted AI agent that connects to any OpenAI-compatible LLM (LM Studio, Ollama, etc.) and can autonomously read and write files, run shell commands, manage git, track todos, coordinate sub-tasks, and much more all from an interactive terminal REPL or a polished web UI.

---

## Security & Sandboxing

LMAgent runs all shell and git commands inside a hardened Docker container. This means the LLM is physically isolated from your host machine it cannot access your files, system settings, or other directories outside the designated workspace folder.

### How it works

A single persistent Docker container is created on first use and reused across sessions. The container is locked down with the following protections:

- **Read-only filesystem** — the container's own system files cannot be modified
- **Workspace-only write access** — the only folder the LLM can read or write is your designated workspace, mounted at `/workspace` inside the container
- **All Linux capabilities dropped** — no privileged operations
- **No privilege escalation** — `no-new-privileges` enforced
- **PID limit** — fork bombs are capped at 128 processes
- **Memory cap** — 512 MB by default (configurable up to 2 GB)
- **CPU cap** — 90% of one core maximum
- **Git sandboxed** — git commands run inside the same Docker container, not on your host. This prevents malicious git hooks or submodule URLs from touching your host machine.

### The workspace folder

Your workspace folder on Windows (or macOS/Linux) is mounted directly into the container. This is a live two-way link files you drop into the folder instantly appear inside the container, and files the LLM creates instantly appear in your folder. The container itself is disposable; your workspace folder persists independently.

The LLM can only see and touch that one folder on your real machine. Everything else on your PC is completely invisible to it.

### Docker network

By default the container runs with `bridge` network mode, meaning it can make outbound internet connections (required for pip installs, web requests, etc.). If you want full air-gap isolation and don't need internet access inside the container, set `DOCKER_NETWORK = "none"` in `sandboxed_shell.py`. This blocks all external network access from inside the container.

### Fallback mode

If Docker is not running, LMAgent falls back to a process-group sandbox with memory and CPU limits. A loud warning is printed to the terminal when this happens. Start Docker Desktop to restore full isolation.

### Requirements

```bash
pip install psutil docker
```

Docker Desktop must be running on Windows. On macOS/Linux, Docker is optional the process-group backend is used by default unless you set `FORCE_DOCKER = True` in `sandboxed_shell.py`.

---

## Files

| File | Role |
|---|---|
| `agent_core.py` | Foundation layer — config, logging, sessions, state, todos, plans, shell sessions, MCP |
| `agent_tools.py` | Tool handlers, LLM client, tool registry, system prompts |
| `agent_main.py` | CLI entrypoint, `run_agent()`, interactive REPL, background scheduler |
| `agent_web.py` | Flask web UI — place next to the three core files |
| `sandboxed_shell.py` | Cross-platform sandboxed subprocess execution (Docker + fallback) |

All five files must be in the same directory.

---

## First Setup

### 1. Install Python dependencies

LMAgent requires Python 3.10 or later.

```bash
pip install requests flask colorama psutil docker
```

### 2. Install and start Docker Desktop

Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/). Make sure it is running before you launch LMAgent. The sandbox container is created automatically on first use.

### 3. Set up your LLM

LMAgent connects to any OpenAI-compatible API endpoint. The easiest option is [LM Studio](https://lmstudio.ai/):

1. Download and install LM Studio
2. Download a model (recommended: a 7B+ instruct or coder model)
3. Go to **Local Server** in LM Studio and click **Start Server**
4. The server runs at `http://localhost:1234` by default — this is what LMAgent expects out of the box

You can also use Ollama, a remote OpenAI-compatible API, or any other compatible server just update `LLM_URL` in your config.

### 4. Create a workspace

Your workspace is the directory LMAgent works inside. It reads and writes files here, and stores all session data in a hidden `.lmagent/` subfolder.

```bash
mkdir ~/lmagent_workspace
```

You can use any existing project directory as your workspace too. On Windows, you can just create a regular folder anywhere drop files into it and the LLM will be able to see and work with them instantly.

### 5. Create a `.env` file

Create a file called `.env` in the same directory as the agent scripts. This is where all your configuration lives:

```env
WORKSPACE="/home/you/lmagent_workspace"
LLM_URL="http://localhost:1234/v1/chat/completions"
LLM_API_KEY="lm-studio"
LLM_MODEL=""
PERMISSION_MODE="normal"
```

- **`WORKSPACE`** — the directory the agent reads/writes files in. Use an absolute path.
- **`LLM_URL`** — your LLM server endpoint. Leave as-is if using LM Studio defaults.
- **`LLM_API_KEY`** — can be anything for local servers; only matters for remote APIs.
- **`LLM_MODEL`** — the model name to request. Leave blank to use whatever is loaded in LM Studio. Set it explicitly if your server hosts multiple models.
- **`PERMISSION_MODE`** — `normal` (asks before destructive actions), `auto` (no prompts), or `manual` (asks before everything).

If you skip this step entirely, LMAgent will prompt you to enter a workspace path on first launch and offer to save it to `.env` automatically.

### 6. Run it

**Terminal REPL:**
```bash
python agent_main.py
```

**Web UI:**
```bash
python agent_web.py
```
Then open `http://localhost:5000` in your browser.

You should see the LMAgent banner and a confirmation that the LLM connected. Type a task and press Enter.

---

## Usage Modes

### Interactive REPL

```bash
python agent_main.py
```

Starts a conversational loop with a background scheduler running alongside it. Type your task and press Enter. The scheduler automatically wakes any sessions that were scheduled to resume at a future time.

### One-shot task

```bash
python agent_main.py "Summarise all Python files in this workspace"
```

Runs a single task and exits with a meaningful exit code (`0` = done, `2` = max iterations hit, `130` = interrupted, `1` = error).

### Web UI

```bash
python agent_web.py
# Custom port:
python agent_web.py 8080
```

Opens at `http://localhost:5000`. Streams tokens live, shows tool calls inline, and lets you browse session history. Also accessible from your phone on the local network — the startup output shows your LAN address.

### Resume a previous session

```bash
python agent_main.py --list-sessions
python agent_main.py --resume SESSION_ID
```

### Plan before acting

```bash
python agent_main.py --plan "refactor the authentication module"
```

The agent produces a step-by-step plan for your approval before it does anything.

### Scheduler daemon only

```bash
python agent_main.py --scheduler
```

Runs only the background scheduler — useful when you're driving LMAgent entirely from the web UI.

### Submit a task to a running scheduler

```bash
python agent_main.py --submit "generate the weekly report"
```

Drops the task into the inbox. A running scheduler picks it up automatically within its next poll interval.

---

## All Configuration Options

Set any of these in your `.env` file or as environment variables.

| Variable | Default | Description |
|---|---|---|
| `WORKSPACE` | `~/lm_workspace` | Directory the agent reads/writes files in |
| `LLM_URL` | `http://localhost:1234/v1/chat/completions` | LLM API endpoint |
| `LLM_API_KEY` | `lm-studio` | API key (anything works for local servers) |
| `LLM_MODEL` | *(blank)* | Model name to request; blank = server default |
| `PERMISSION_MODE` | `normal` | `auto` / `normal` / `manual` |
| `MAX_ITERATIONS` | `150` | Hard cap on iterations per task |
| `TEMPERATURE` | `0.9` | LLM sampling temperature |
| `LLM_TIMEOUT` | `560` | Seconds to wait for an LLM response |
| `THINKING_MODEL` | `true` | Strip `<think>` blocks (for QwQ, DeepSeek-R1, etc.) |
| `SUMMARIZATION_THRESHOLD` | `80000` | Token count at which context gets compacted |
| `ENABLE_SUB_AGENTS` | `true` | Allow the agent to spawn sub-agents for file creation |
| `ENABLE_TODO_TRACKING` | `true` | Track todos across iterations |
| `ENABLE_PLAN_ENFORCEMENT` | `true` | Enforce step-by-step plans |
| `SCHEDULER_POLL_INTERVAL` | `60` | Seconds between scheduler wake-up checks |

---

## Available Tools

LMAgent calls these tools autonomously during a task.

**Files**
- `read` — read a file
- `write` — create or overwrite a file
- `edit` — fuzzy search-and-replace within a file
- `glob` — find files matching a pattern
- `grep` — search text across files
- `ls` — list a directory
- `mkdir` — create a directory

**Shell**
- `shell` — run a command inside the Docker sandbox (bash on Linux/macOS, cmd on Windows); falls back to process-group isolation if Docker is unavailable

**Git**
- `git_status`, `git_diff`, `git_add`, `git_commit`, `git_branch` — all git operations run inside the Docker sandbox, not on your host machine

**Task management**
- `todo_add`, `todo_complete`, `todo_update`, `todo_list`
- `plan_complete_step`
- `task_state_update`, `task_state_get`, `task_reconcile`

**Sub-agents**
- `task` — delegates single-file creation to an isolated sub-agent (file tools only, no shell or git access)

**Utilities**
- `get_time` — current date and time

---

## REPL Slash Commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/sessions` | List recent sessions |
| `/mode <level>` | Change permission mode (`auto` / `normal` / `manual`) |
| `/plan` | Show the current execution plan |
| `/todo` | Show the current todo list |
| `/status` | Show agent, LLM, and MCP status |
| `/soul` | Show the loaded personality config |
| `/new` | Start a completely fresh session |
| `/session` | Show the current session ID |
| `quit` | Exit |

---

## Personality (Soul)

Create a `.soul.md` file in your workspace to give LMAgent a custom personality or standing instructions that apply to every session:

```markdown
# My Agent

You are a meticulous backend engineer who prefers functional patterns.
Always add type hints. Write tests before implementation.
When uncertain, ask rather than assume.
```

---

## Project Config

Create a `.lmagent.md` file in your workspace to inject persistent project context — tech stack, coding conventions, file layout, anything the agent should always know:

```markdown
# Project: MyApp

Stack: Python 3.12, FastAPI, PostgreSQL, Redis
Tests: pytest, always run before committing
Style: Black formatter, max line length 100
Main source: src/
```

---

## MCP Servers

LMAgent supports the [Model Context Protocol](https://modelcontextprotocol.io/) for connecting external tool servers, giving the agent access to things like web search, databases, browser control, and more.

### Step 1 — Install an MCP server

Most MCP servers are distributed via npm or pip. You need [Node.js](https://nodejs.org/) installed for npm-based servers.

**npm-based servers (most common):**
```bash
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-brave-search
npm install -g @modelcontextprotocol/server-puppeteer
```

**pip-based servers:**
```bash
pip install mcp-server-fetch
```

You can browse available servers at the [MCP server registry](https://github.com/modelcontextprotocol/servers).

### Step 2 — Create the config file

Create `.lmagent/mcp.json` inside your workspace directory:

```bash
mkdir -p ~/lmagent_workspace/.lmagent
```

Then create the file `~/lmagent_workspace/.lmagent/mcp.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/your/path"],
      "env": {}
    }
  }
}
```

### Step 3 — Add more servers

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/you/lmagent_workspace"],
      "env": {}
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-api-key-here"
      }
    },
    "fetch": {
      "command": "python",
      "args": ["-m", "mcp_server_fetch"],
      "env": {}
    }
  }
}
```

### Step 4 — Restart LMAgent

MCP servers are loaded at startup. Restart `agent_main.py` or `agent_web.py` and you'll see a confirmation line for each server that connected successfully:

```
[INFO] MCP 'filesystem' started
[INFO] MCP 'brave-search' started
[INFO] Loaded 2 MCP servers
```

MCP tools appear automatically alongside built-in tools. In the web UI you can see them in the **Tools** panel under their own MCP section.

### Troubleshooting MCP

- **Server won't start** — run the command manually in your terminal first to check for errors. Common causes are Node.js not installed, a missing API key in `env`, or a wrong path.
- **Tools not showing up** — check that the server name in `mcp.json` has no typos and that LMAgent restarted after you edited the file.
- **npm not found** — install Node.js from [nodejs.org](https://nodejs.org/). The LTS version is recommended.

---

## Scheduled Waits

The agent can suspend itself until a future time by outputting a special token:

```
WAIT: 2026-03-01T09:00:00: Waiting for market open.
```

The background scheduler wakes the session automatically when the time arrives. Waiting sessions appear in the web UI session browser and can also be manually resumed with `--resume`.

---

## Session Storage

Every task creates a session stored under `<workspace>/.lmagent/sessions/`. Sessions survive restarts and can be resumed, listed, and browsed from the REPL and the web UI.

```bash
python agent_main.py --list-sessions
python agent_main.py --resume 20260220_143512_a1b2c3
```

---

## Web UI Features

- Live token streaming with full markdown rendering
- Inline tool call display (pending → success / failure)
- Session browser with status indicators
- Tool explorer showing all built-in and MCP tools with their parameters
- Slash commands (`/help`, `/mode`, `/todo`, `/plan`, `/soul`, `/status`, `/session`)
- Stop button to interrupt a running task mid-stream
- New button to clear and start a fresh session
- Chat history replay on page refresh
- Mobile-friendly — accessible from any device on your local network

---

## Tips

- **First run slow?** The agent validates the LLM connection at startup. If it fails, check that your LM Studio server is running and that `LLM_URL` in `.env` is correct. The Docker container is also created on first use which may take a moment to pull the image.
- **Docker not running?** LMAgent will fall back to a process-group sandbox and print a warning. Start Docker Desktop to restore full isolation.
- **Stuck agent?** The loop detector intervenes after repeated identical tool calls, consecutive errors, or empty iterations. You can also click Stop in the web UI at any time.
- **Large projects?** The context window compacts automatically once it exceeds `SUMMARIZATION_THRESHOLD` tokens, preserving task-critical messages and summarising older ones.
- **Thinking models (QwQ, DeepSeek-R1)?** `THINKING_MODEL=true` is on by default — reasoning tokens are stripped from the context window before being fed back to the model.
- **Multiple sessions at once?** The scheduler, REPL, and web UI can all run simultaneously. Each agent session gets its own thread-local shell process so they never interfere with each other.
- **Want full air-gap isolation?** Set `DOCKER_NETWORK = "none"` in `sandboxed_shell.py` to block all outbound network access from inside the container. Note this will also disable pip installs and any web requests from within the sandbox.
