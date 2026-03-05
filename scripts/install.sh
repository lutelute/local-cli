#!/usr/bin/env bash
#
# install.sh - Installation script for local-cli AI coding agent.
#
# Copies the local-cli scripts and Python package to standard locations:
#   - ~/.local/bin/local-cli        (launcher script)
#   - ~/.local/lib/local-cli/       (Python package)
#   - ~/.config/local-cli/          (configuration directory)
#   - ~/.local/state/local-cli/     (state directory)
#
# Adds ~/.local/bin to PATH in .bashrc/.zshrc if needed.
#
# Compatible with bash 3.2+ (macOS default).
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

readonly BIN_DIR="${HOME}/.local/bin"
readonly LIB_DIR="${HOME}/.local/lib/local-cli"
readonly CONFIG_DIR="${HOME}/.config/local-cli"
readonly STATE_DIR="${HOME}/.local/state/local-cli"

# ---------------------------------------------------------------------------
# Logging helpers (same style as local-cli.sh)
# ---------------------------------------------------------------------------

info() {
    printf '[install] %s\n' "$1" >&2
}

warn() {
    printf '[install] WARNING: %s\n' "$1" >&2
}

error_exit() {
    printf '[install] ERROR: %s\n' "$1" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------

resolve_project_root() {
    # Determine the project root relative to this script's location.
    local script_dir
    script_dir=$(cd "$(dirname "$0")" && pwd)
    local project_dir
    project_dir=$(cd "${script_dir}/.." && pwd)
    printf '%s' "$project_dir"
}

# ---------------------------------------------------------------------------
# Validate source files exist
# ---------------------------------------------------------------------------

validate_sources() {
    local project_dir="$1"

    if [ ! -f "${project_dir}/scripts/local-cli.sh" ]; then
        error_exit "Cannot find scripts/local-cli.sh in ${project_dir}. Run this script from the project root."
    fi

    if [ ! -d "${project_dir}/local_cli" ]; then
        error_exit "Cannot find local_cli/ directory in ${project_dir}. Run this script from the project root."
    fi

    if [ ! -f "${project_dir}/local_cli/__init__.py" ]; then
        error_exit "Cannot find local_cli/__init__.py in ${project_dir}. Package appears incomplete."
    fi
}

# ---------------------------------------------------------------------------
# Create directories
# ---------------------------------------------------------------------------

create_directories() {
    info "Creating directories..."

    mkdir -p "$BIN_DIR"
    info "  ${BIN_DIR}"

    mkdir -p "$LIB_DIR"
    info "  ${LIB_DIR}"

    mkdir -p "$CONFIG_DIR"
    info "  ${CONFIG_DIR}"

    mkdir -p "$STATE_DIR"
    info "  ${STATE_DIR}"
}

# ---------------------------------------------------------------------------
# Copy files
# ---------------------------------------------------------------------------

install_launcher() {
    local project_dir="$1"

    info "Installing launcher script..."

    # Copy the launcher script to ~/.local/bin/local-cli.
    cp "${project_dir}/scripts/local-cli.sh" "${BIN_DIR}/local-cli"
    chmod +x "${BIN_DIR}/local-cli"

    info "  ${BIN_DIR}/local-cli"
}

install_library() {
    local project_dir="$1"

    info "Installing Python package..."

    # Remove previous installation if present.
    if [ -d "$LIB_DIR" ]; then
        rm -rf "$LIB_DIR"
        mkdir -p "$LIB_DIR"
    fi

    # Copy the entire local_cli/ package.
    cp -R "${project_dir}/local_cli" "${LIB_DIR}/local_cli"

    # Remove __pycache__ directories from the installed copy.
    find "$LIB_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    info "  ${LIB_DIR}/local_cli/"
}

create_default_config() {
    # Create a default config file if one does not already exist.
    local config_file="${CONFIG_DIR}/config"

    if [ -f "$config_file" ]; then
        info "Config file already exists: ${config_file} (not overwritten)"
        return
    fi

    info "Creating default config file..."

    cat > "$config_file" <<'CONFIGEOF'
# local-cli configuration
# Priority: CLI args > env vars > this file > defaults
#
# model=qwen3:8b
# sidecar_model=qwen3:1.7b
# ollama_host=http://localhost:11434
# debug=false
CONFIGEOF

    info "  ${config_file}"
}

# ---------------------------------------------------------------------------
# Update PATH in shell RC files
# ---------------------------------------------------------------------------

add_to_path() {
    # Add ~/.local/bin to PATH in shell RC files if not already present.

    local path_line='export PATH="${HOME}/.local/bin:${PATH}"'
    local updated=0

    # Check if ~/.local/bin is already in PATH.
    case ":${PATH}:" in
        *":${BIN_DIR}:"*)
            info "${BIN_DIR} is already in PATH."
            return
            ;;
    esac

    # Update .bashrc if it exists or if bash is the default shell.
    local bashrc="${HOME}/.bashrc"
    if [ -f "$bashrc" ]; then
        if ! grep -qF '/.local/bin' "$bashrc" 2>/dev/null; then
            printf '\n# Added by local-cli installer\n%s\n' "$path_line" >> "$bashrc"
            info "Added ${BIN_DIR} to PATH in ${bashrc}"
            updated=1
        fi
    fi

    # Update .zshrc if it exists or if zsh is the default shell.
    local zshrc="${HOME}/.zshrc"
    if [ -f "$zshrc" ]; then
        if ! grep -qF '/.local/bin' "$zshrc" 2>/dev/null; then
            printf '\n# Added by local-cli installer\n%s\n' "$path_line" >> "$zshrc"
            info "Added ${BIN_DIR} to PATH in ${zshrc}"
            updated=1
        fi
    fi

    # If neither RC file existed and PATH doesn't include our dir,
    # create .bashrc with the PATH addition.
    if [ "$updated" -eq 0 ]; then
        if [ ! -f "$bashrc" ] && [ ! -f "$zshrc" ]; then
            printf '# Added by local-cli installer\n%s\n' "$path_line" >> "$bashrc"
            info "Created ${bashrc} with ${BIN_DIR} in PATH"
            updated=1
        fi
    fi

    if [ "$updated" -eq 1 ]; then
        warn "Restart your shell or run 'source ~/.bashrc' (or ~/.zshrc) to update PATH."
    fi
}

# ---------------------------------------------------------------------------
# Print post-install instructions
# ---------------------------------------------------------------------------

print_instructions() {
    printf '\n'
    printf '=%.0s' {1..60}
    printf '\n'
    printf '  local-cli installed successfully!\n'
    printf '=%.0s' {1..60}
    printf '\n\n'
    printf 'Installed files:\n'
    printf '  Launcher:  %s\n' "${BIN_DIR}/local-cli"
    printf '  Library:   %s\n' "${LIB_DIR}/local_cli/"
    printf '  Config:    %s\n' "${CONFIG_DIR}/"
    printf '  State:     %s\n' "${STATE_DIR}/"
    printf '\n'
    printf 'Prerequisites:\n'
    printf '  1. Install Ollama:  https://ollama.com/\n'
    printf '  2. Pull a model:    ollama pull qwen3:8b\n'
    printf '\n'
    printf 'Usage:\n'
    printf '  local-cli                    # Start interactive session\n'
    printf '  local-cli --model qwen3:8b   # Use specific model\n'
    printf '  local-cli --debug            # Enable debug output\n'
    printf '\n'
    printf 'Configuration:\n'
    printf '  Edit %s/config\n' "${CONFIG_DIR}"
    printf '\n'
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    info "Installing local-cli..."

    # 1. Resolve project root.
    local project_dir
    project_dir=$(resolve_project_root)
    info "Project root: ${project_dir}"

    # 2. Validate source files exist.
    validate_sources "$project_dir"

    # 3. Create directories.
    create_directories

    # 4. Install launcher script.
    install_launcher "$project_dir"

    # 5. Install Python package.
    install_library "$project_dir"

    # 6. Create default config (if not present).
    create_default_config

    # 7. Add ~/.local/bin to PATH if needed.
    add_to_path

    # 8. Print post-install instructions.
    print_instructions

    info "Done."
}

main "$@"
