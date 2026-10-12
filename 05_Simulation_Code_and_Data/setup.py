#!/bin/bash
# setup.sh - Complete environment setup

echo "=== Setting up IoT Healthcare Simulation Environment ==="

# 1. Create venv
echo "Creating virtual environment..."
python3 -m venv venv

# 2. Activate
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Install packages
echo "Installing required packages..."
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scipy pyarrow fastparquet

# 4. Verify
echo "Verifying installation..."
python -c "import pandas as pd; import numpy as np; print('✓ Pandas:', pd.__version__); print('✓ NumPy:', np.__version__)"

echo "=== Setup Complete ==="
echo "Run: source venv/bin/activate"
echo "Then: python run_batch.py Controlled 0"
