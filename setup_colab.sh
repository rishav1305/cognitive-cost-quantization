#!/usr/bin/env bash
# setup_colab.sh — One-command Colab provisioning
# Run this after SSHing into Colab to set up the full research environment.
#
# Prerequisites:
#   - Colab Secrets set: HF_TOKEN, ANTHROPIC_AUTH_TOKEN, OPENAI_API_KEY
#   - setup-claude.js SCP'd to home directory (not in repo — corporate script)
#
# Usage: ./setup_colab.sh

set -euo pipefail

echo "=== Cognitive Cost Quantization — Colab Setup ==="
echo ""

# ---- 1. Verify GPU ----
echo "[1/8] Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: No GPU detected. Benchmarks will not work."
fi

# ---- 2. Install Node.js 20 LTS (for Claude Code + Codex) ----
echo "[2/8] Installing Node.js 20 LTS..."
if ! command -v node &>/dev/null || [[ "$(node -v)" != v20* ]]; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
echo "Node.js $(node -v)"

# ---- 3. Install OpenAI Codex CLI ----
echo "[3/8] Installing Codex CLI..."
if ! command -v codex &>/dev/null; then
    npm install -g @openai/codex
fi
# Set API key from Colab Secret
if [ -n "${OPENAI_API_KEY:-}" ]; then
    export OPENAI_API_KEY
    echo "OPENAI_API_KEY set from environment"
else
    echo "WARNING: OPENAI_API_KEY not set. Codex CLI won't work."
fi

# ---- 4. Pre-write Claude Code settings (Mercury auth) ----
echo "[4/8] Configuring Claude Code auth..."
mkdir -p ~/.claude
if [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
    cat > ~/.claude/settings.json <<SETTINGS
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "${ANTHROPIC_AUTH_TOKEN}",
    "ANTHROPIC_BASE_URL": "https://api.mercury.weather.com/litellm"
  }
}
SETTINGS
    echo "Claude Code settings written"
else
    echo "WARNING: ANTHROPIC_AUTH_TOKEN not set. Claude Code auth will fail."
fi

# ---- 5. Install Claude Code ----
echo "[5/8] Installing Claude Code..."
if ! command -v claude &>/dev/null; then
    if [ -f ~/setup-claude.js ]; then
        node ~/setup-claude.js
    else
        curl -fsSL https://claude.ai/install.sh | bash
    fi
fi
echo "Claude Code: $(claude --version 2>/dev/null || echo 'not installed')"

# ---- 6. Install Python dependencies ----
echo "[6/8] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -e .
echo "Python deps installed"

# ---- 7. HuggingFace login ----
echo "[7/8] Logging into HuggingFace..."
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "HuggingFace login successful"
else
    echo "WARNING: HF_TOKEN not set. Model downloads will fail for gated models."
fi

# ---- 8. Clone/pull research repo ----
echo "[8/8] Setting up research repo..."
REPO_DIR=~/cognitive-cost-quantization
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR" && git pull --rebase
    echo "Repo updated"
else
    git clone https://github.com/rishav1305/cognitive-cost-quantization.git "$REPO_DIR"
    echo "Repo cloned"
fi
cd "$REPO_DIR"

echo ""
echo "=== Setup Complete ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Python: $(python --version)"
echo "Repo: $REPO_DIR"
echo ""
echo "Next steps:"
echo "  python -m src.benchmark --model meta-llama/Llama-3.2-3B --tasks arc_easy --limit 10"
echo "  claude   # start Claude Code"
