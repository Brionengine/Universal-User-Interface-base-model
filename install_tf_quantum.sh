#!/bin/bash
# install_tf_quantum.sh
# Script to easily download, build, and install TensorFlow Quantum
# Designed for Brion Quantum AI integration

set -e

echo "=================================================================="
echo "Brion Quantum AI - TensorFlow Quantum (TFQ) Installer"
echo "=================================================================="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

echo "[*] Checking system dependencies..."
# Update this list based on specific build requirements if building from source
if grep -q "Kali" /etc/os-release; then
    echo "[+] Kali Linux detected. Installing build dependencies..."
    # sudo apt-get update
    # sudo apt-get install -y build-essential bazel-bootstrap libssl-dev
fi

echo "[*] Creating virtual environment 'tfq_env'..."
# Try to find a compatible python version (3.9 - 3.11)
PYTHON_CMD="python3"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
fi

echo "Using Python: $PYTHON_CMD"
if ! $PYTHON_CMD -m venv tfq_env; then
    echo "Error: Failed to create virtual environment with $PYTHON_CMD"
    echo "Please install Python 3.10 or 3.11: sudo apt-get install python3.11-venv"
    exit 1
fi
source tfq_env/bin/activate

echo "[*] Installing TensorFlow 2.14.0 (Compatible with TFQ 0.7.3+)..."
pip install --upgrade pip
pip install tensorflow==2.14.0

echo "[*] Checking for local TensorFlow Quantum wheel..."
WHEEL_FILE=$(find . -maxdepth 1 -name "tensorflow_quantum-*.whl" | head -n 1)

if [ -n "$WHEEL_FILE" ]; then
    echo "[+] Found local wheel: $WHEEL_FILE"
    echo "[*] Installing from local wheel..."
    pip install "$WHEEL_FILE"
    echo "[+] TensorFlow Quantum installed from local wheel."
else
    echo "[!] No local wheel found in current directory."
    echo "[*] Attempting to install TensorFlow Quantum via pip (PyPI)..."
    if pip install tensorflow-quantum; then
        echo "[+] TensorFlow Quantum installed successfully from PyPI."
    else
        echo "[!] PyPI installation failed."
        exit 1
    fi
fi

echo "[*] Verifying installation..."
python3 -c "import tensorflow as tf; import tensorflow_quantum as tfq; print(f'TF Version: {tf.__version__}'); print(f'TFQ Version: {tfq.__version__}')"

echo "=================================================================="
echo "Installation Complete."
echo "Activate environment with: source tfq_env/bin/activate"
echo "=================================================================="

