#!/bin/bash

# setup_kali_env.sh
# Script to setup the Quantum Mining environment on Kali Linux
# Author: Brionengine

echo "=================================================================="
echo "Initializing Quantum TPU HPC Mining Environment on Kali Linux"
echo "=================================================================="

# Check if running on Kali
if grep -q "Kali" /etc/os-release; then
    echo "[+] Detected Kali Linux"
else
    echo "[!] Warning: Not running on Kali Linux. Some tools may be missing."
fi

echo "[*] Updating system repositories..."
# sudo apt-get update

echo "[*] Installing system dependencies..."
# sudo apt-get install -y python3-pip python3-venv build-essential libssl-dev libffi-dev

echo "[*] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[*] Installing Python dependencies..."
pip install -r requirements.txt

echo "[*] Installing TensorFlow Quantum dependencies..."
# Note: TensorFlow Quantum requires specific Linux setup
pip install --upgrade pip
pip install tensorflow==2.14.0

echo "[*] Setting up Deepseek Agent configuration..."
export DEEPSEEK_MODEL="3.2-speciale"

echo "[+] Environment setup complete!"
echo "To run the miner:"
echo "  source venv/bin/activate"
echo "  python3 quantum_mining_main.py"

