#!/bin/bash
# Linux VM (Ubuntu 22.04/24.04) setup for MLIP SQS relaxation pipeline.
# Run as a user with sudo rights.

set -euo pipefail

# 1. System packages
sudo apt-get update -q
sudo apt-get install -y -q \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    cmake \
    libopenmpi-dev     # optional, for MPI parallelism later

# 2. Python virtual environment
python3 -m venv ~/.venv/mlip
source ~/.venv/mlip/bin/activate

# 3. Upgrade pip and install core scientific stack
pip install -q --upgrade pip setuptools wheel
pip install -q numpy scipy pandas

# 4. Install materials/MLIP packages
#    icet compiles C++ extensions; python3-dev is required.
pip install -q ase icet mace-torch

# 5. Verify imports
python - <<'PY'
import ase, icet, mace
print(f"ase {ase.__version__}, icet {icet.__version__}, mace OK")
PY

echo "Setup complete. Activate with: source ~/.venv/mlip/bin/activate"
