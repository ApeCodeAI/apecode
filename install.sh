#!/usr/bin/env bash
set -euo pipefail

install_uv() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://astral.sh/uv/install.sh | sh
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
    return
  fi

  echo "Error: curl or wget is required to install uv." >&2
  exit 1
}

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  install_uv
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found after installation." >&2
  exit 1
fi

uv tool install --python 3.13 apecode
