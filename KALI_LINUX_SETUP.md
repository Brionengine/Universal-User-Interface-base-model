# Kali Linux Setup Guide for the Universal User Interface Base Model

This guide explains how to run and validate the Universal User Interface (UUI) base model on a Kali Linux environment (or compatible Debian-based security distribution).

## Prerequisites

- Kali Linux (Rolling edition recommended)
- Python 3.10+
- Internet connection for Python package downloads

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Brionengine/Universal-User-Interface-base-model.git
    cd Universal-User-Interface-base-model
    ```

2.  **Run Setup Script**
    The repository includes a Kali-aware setup script that validates the OS, installs dependencies, and prepares a virtual environment:
    ```bash
    chmod +x setup_kali_env.sh
    ./setup_kali_env.sh
    ```
    The script will warn if you are not on Kali and pause briefly to let you abort.

## Verifying the Setup

From inside the repository and with the virtual environment activated, you can run a quick check to confirm everything is wired up:

```bash
source venv/bin/activate
python3 - <<'PYCODE'
from UUI import UUI
ui = UUI()
print(ui.secure_session())
print(ui.set_security_headers())
PYCODE
```

You should see confirmation that a secure session has started and headers were applied.

## Deepseek Integration

The base model includes optional integration with **Deepseek V3.2 Speciale** for advanced agentic reasoning.
To enable full API access:

1.  Obtain a Deepseek API key.
2.  Export it in your shell before running the model:
    ```bash
    export DEEPSEEK_API_KEY="your_key_here"
    ```

## Troubleshooting

-   **Not on Kali:** The setup script will warn if it does not detect Kali via `/etc/os-release`. You can still continue, but some tools may be missing.
-   **TensorFlow Installation:** TensorFlow is optional. Comment out the line in `setup_kali_env.sh` if you do not need it or if you are on hardware without compatible acceleration.
-   **Virtual Environment Issues:** Remove the `venv/` directory and rerun the setup script to rebuild a clean environment.

