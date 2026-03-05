#!/usr/bin/env bash
#
# local-cli.sh - Launcher script for local-cli AI coding agent.
#
# Validates prerequisites (Python 3, Ollama), reads configuration,
# ensures the requested model is available, and launches the Python
# application.
#
# Compatible with bash 3.2+ (macOS default).
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

readonly STATE_DIR="${HOME}/.local/state/local-cli"
readonly CONFIG_FILE="${HOME}/.config/local-cli/config"
readonly DEFAULT_OLLAMA_HOST="http://localhost:11434"
readonly DEFAULT_MODEL="qwen3:8b"
readonly RETRY_ATTEMPTS=15
readonly RETRY_INTERVAL=2

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

info() {
    printf '[local-cli] %s\n' "$1" >&2
}

warn() {
    printf '[local-cli] WARNING: %s\n' "$1" >&2
}

error_exit() {
    printf '[local-cli] ERROR: %s\n' "$1" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Read config file (grep/cut, never source)
# ---------------------------------------------------------------------------

read_config_value() {
    # Read a key=value pair from the config file.
    # Args: $1 = key name, $2 = default value
    local key="$1"
    local default_value="$2"

    if [ ! -f "$CONFIG_FILE" ]; then
        printf '%s' "$default_value"
        return
    fi

    # Reject symlinks.
    if [ -L "$CONFIG_FILE" ]; then
        printf '%s' "$default_value"
        return
    fi

    # grep for the key, cut the value after the first '='.
    # Use head -1 to take only the first match.
    local value
    value=$(grep "^${key}=" "$CONFIG_FILE" 2>/dev/null | head -1 | cut -d'=' -f2-)

    if [ -n "$value" ]; then
        # Trim leading/trailing whitespace (bash 3.2 compatible).
        value=$(printf '%s' "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        printf '%s' "$value"
    else
        printf '%s' "$default_value"
    fi
}

# ---------------------------------------------------------------------------
# Validate OLLAMA_HOST
# ---------------------------------------------------------------------------

validate_ollama_host() {
    # Validate that OLLAMA_HOST points to localhost and has no @ symbol.
    # Args: $1 = the host URL
    local host_url="$1"

    # Reject empty URL.
    if [ -z "$host_url" ]; then
        error_exit "OLLAMA_HOST is empty."
    fi

    # Reject URLs containing @ (credential injection risk).
    case "$host_url" in
        *@*)
            error_exit "OLLAMA_HOST must not contain '@' symbol: ${host_url}"
            ;;
    esac

    # Extract hostname from URL.
    # Remove scheme (http:// or https://).
    local stripped
    stripped=$(printf '%s' "$host_url" | sed 's|^https\{0,1\}://||')

    # Remove path component.
    stripped=$(printf '%s' "$stripped" | cut -d'/' -f1)

    # Remove port component.
    local hostname
    hostname=$(printf '%s' "$stripped" | sed 's/:[0-9]*$//')

    # Validate against known localhost addresses.
    case "$hostname" in
        localhost|127.0.0.1|"::1"|"[::1]"|0.0.0.0)
            # Valid localhost address.
            return 0
            ;;
        *)
            error_exit "OLLAMA_HOST must point to localhost, got: ${host_url} (hostname: ${hostname})"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Check Python 3
# ---------------------------------------------------------------------------

check_python() {
    if command -v python3 >/dev/null 2>&1; then
        return 0
    fi

    error_exit "Python 3 is required but not found. Install it from https://www.python.org/ or via your package manager."
}

# ---------------------------------------------------------------------------
# Detect OS
# ---------------------------------------------------------------------------

detect_os() {
    # Prints "macos" or "linux".
    case "$(uname -s)" in
        Darwin*)
            printf 'macos'
            ;;
        Linux*)
            printf 'linux'
            ;;
        *)
            printf 'unknown'
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Check if Ollama is running
# ---------------------------------------------------------------------------

check_ollama_running() {
    # Check if Ollama is reachable by querying /api/tags.
    # Args: $1 = ollama host URL
    # Returns 0 if running, 1 if not.
    local host_url="$1"
    if curl -s --max-time 5 "${host_url}/api/tags" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Auto-start Ollama
# ---------------------------------------------------------------------------

start_ollama() {
    # Attempt to start Ollama based on the detected OS.
    local os_type
    os_type=$(detect_os)

    case "$os_type" in
        macos)
            info "Starting Ollama via 'open -a Ollama'..."
            open -a Ollama 2>/dev/null || true
            ;;
        linux)
            info "Starting Ollama via 'ollama serve'..."
            if command -v ollama >/dev/null 2>&1; then
                ollama serve >/dev/null 2>&1 &
            else
                error_exit "Ollama is not installed. Install it from https://ollama.com/"
            fi
            ;;
        *)
            error_exit "Unsupported OS. Cannot auto-start Ollama."
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Wait for Ollama with retry loop
# ---------------------------------------------------------------------------

wait_for_ollama() {
    # Wait for Ollama to become available with retries.
    # Args: $1 = ollama host URL
    local host_url="$1"
    local attempt=1

    while [ "$attempt" -le "$RETRY_ATTEMPTS" ]; do
        if check_ollama_running "$host_url"; then
            info "Ollama is running."
            return 0
        fi
        info "Waiting for Ollama... (attempt ${attempt}/${RETRY_ATTEMPTS})"
        sleep "$RETRY_INTERVAL"
        attempt=$((attempt + 1))
    done

    error_exit "Ollama did not start after ${RETRY_ATTEMPTS} attempts (${RETRY_INTERVAL}s intervals). Is Ollama installed? Visit https://ollama.com/"
}

# ---------------------------------------------------------------------------
# Validate model availability
# ---------------------------------------------------------------------------

validate_model() {
    # Check if the requested model exists on the Ollama server.
    # If missing, offer to pull it.
    # Args: $1 = ollama host URL, $2 = model name
    local host_url="$1"
    local model="$2"

    # Query available models.
    local tags_json
    tags_json=$(curl -s --max-time 10 "${host_url}/api/tags" 2>/dev/null) || {
        warn "Could not query Ollama for available models."
        return 0
    }

    # Check if the model name appears in the tags response.
    # Use grep to search for the model name in the JSON output.
    # Match both exact name and name without tag (e.g., "qwen3:8b" or "qwen3").
    if printf '%s' "$tags_json" | grep -q "\"name\"[[:space:]]*:[[:space:]]*\"${model}\""; then
        info "Model '${model}' is available."
        return 0
    fi

    # Also check if model appears with a different tag (e.g., model:latest).
    if printf '%s' "$tags_json" | grep -q "\"name\"[[:space:]]*:[[:space:]]*\"${model}:"; then
        info "Model '${model}' is available."
        return 0
    fi

    # Model not found - show available models and offer to pull.
    warn "Model '${model}' is not available on this Ollama server."

    # Extract model names from JSON (simple grep/sed approach).
    local available
    available=$(printf '%s' "$tags_json" | grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/"name"[[:space:]]*:[[:space:]]*"//;s/"$//' | tr '\n' ', ' | sed 's/,$//')

    if [ -n "$available" ]; then
        info "Available models: ${available}"
    fi

    # Offer to pull the model.
    printf '[local-cli] Would you like to pull model "%s"? [y/N] ' "$model" >&2
    local answer
    read -r answer

    case "$answer" in
        [yY]|[yY][eE][sS])
            info "Pulling model '${model}'... (this may take a while)"
            if command -v ollama >/dev/null 2>&1; then
                ollama pull "$model"
            else
                # Fall back to API pull.
                curl -s "${host_url}/api/pull" -d "{\"name\":\"${model}\"}" | while IFS= read -r line; do
                    local status_msg
                    status_msg=$(printf '%s' "$line" | grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/"status"[[:space:]]*:[[:space:]]*"//;s/"$//')
                    if [ -n "$status_msg" ]; then
                        printf '\r[local-cli] %s' "$status_msg" >&2
                    fi
                done
                printf '\n' >&2
            fi
            info "Model '${model}' pull complete."
            ;;
        *)
            warn "Continuing without model '${model}'. The CLI may fail if the model is not available."
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    # 1. Create state directory.
    mkdir -p "$STATE_DIR"

    # 2. Read configuration (config file values, overridden by env vars).
    local cfg_model
    cfg_model=$(read_config_value "model" "$DEFAULT_MODEL")

    local cfg_ollama_host
    cfg_ollama_host=$(read_config_value "ollama_host" "$DEFAULT_OLLAMA_HOST")

    local cfg_sidecar_model
    cfg_sidecar_model=$(read_config_value "sidecar_model" "")

    local cfg_debug
    cfg_debug=$(read_config_value "debug" "")

    # Environment variables override config file values.
    local model="${LOCAL_CLI_MODEL:-$cfg_model}"
    local ollama_host="${OLLAMA_HOST:-$cfg_ollama_host}"
    local sidecar_model="${LOCAL_CLI_SIDECAR_MODEL:-$cfg_sidecar_model}"
    local debug="${LOCAL_CLI_DEBUG:-$cfg_debug}"

    # 3. Validate OLLAMA_HOST (must be localhost, no @ symbol).
    validate_ollama_host "$ollama_host"

    # 4. Check Python 3 availability.
    check_python

    # 5. Detect if Ollama is running.
    if ! check_ollama_running "$ollama_host"; then
        info "Ollama is not running at ${ollama_host}."

        # 6. Auto-start Ollama.
        start_ollama

        # 7. Retry loop - wait for Ollama to become available.
        wait_for_ollama "$ollama_host"
    else
        info "Ollama is running at ${ollama_host}."
    fi

    # 8. Validate the requested model exists.
    validate_model "$ollama_host" "$model"

    # 9. Set environment variables for the Python application.
    export OLLAMA_HOST="$ollama_host"
    export LOCAL_CLI_MODEL="$model"

    if [ -n "$sidecar_model" ]; then
        export LOCAL_CLI_SIDECAR_MODEL="$sidecar_model"
    fi

    if [ -n "$debug" ]; then
        export LOCAL_CLI_DEBUG="$debug"
    fi

    # 10. Resolve the script directory to find the project root.
    local script_dir
    script_dir=$(cd "$(dirname "$0")" && pwd)
    local project_dir
    project_dir=$(cd "${script_dir}/.." && pwd)

    # 11. Launch the Python application.
    info "Starting local-cli (model: ${model})..."
    exec python3 -m local_cli "$@"
}

main "$@"
