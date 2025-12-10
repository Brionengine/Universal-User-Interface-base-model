#!/bin/bash
set -euo pipefail

# setup_kali_env.sh
# Script to prepare the Universal User Interface base model environment on Kali Linux

echo "=================================================================="
echo "Initializing Universal User Interface Environment on Kali Linux"
echo "=================================================================="

# Check if running on Kali
if grep -qi "^ID=kali" /etc/os-release; then
    echo "[+] Detected Kali Linux"
else
    echo "[!] Warning: Not running on Kali Linux. Some tools may be missing."
    echo "[!] Press Ctrl+C to abort or continue at your own risk."
    sleep 3
fi

echo "[*] Updating system repositories..."
sudo apt-get update -y

echo "[*] Installing system dependencies..."
sudo apt-get install -y python3 python3-venv python3-pip build-essential libssl-dev libffi-dev

echo "[*] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[*] Upgrading pip..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "[*] Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "[!] requirements.txt not found. Skipping optional dependency installation."
fi

echo "[*] (Optional) Installing TensorFlow for advanced simulations..."
echo "    You can comment this out if you do not need TensorFlow."
pip install tensorflow==2.14.0

echo "[*] Setting up Deepseek Agent configuration..."
export DEEPSEEK_MODEL="3.2-speciale"

echo "[+] Environment setup complete!"
echo "To start using the base model:"
echo "  source venv/bin/activate"
echo "  python3 - <<'PYCODE'"
echo "from UUI import UUI; ui = UUI(); print(ui.secure_session())"
echo "PYCODE"

