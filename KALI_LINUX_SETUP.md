# Kali Linux Setup Guide for Quantum TPU HPC Mining

This guide explains how to deploy the Quantum TPU HPC Mining system on a Kali Linux environment (or compatible Debian-based security distribution).

## Prerequisites

- Kali Linux (Rolling edition recommended)
- Python 3.10+
- 16GB RAM minimum (for quantum simulation)
- Internet connection for IBM Quantum and Bitcoin Network access

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Brionengine/Quantum-TPU-HPC-Mining.git
    cd Quantum-TPU-HPC-Mining
    ```

2.  **Run Setup Script**
    We have provided a specialized setup script for Kali Linux:
    ```bash
    chmod +x setup_kali_env.sh
    ./setup_kali_env.sh
    ```

## Security & Ethics

As a security-focused distribution, Kali Linux provides tools that should be used responsibly.

-   **Ethical Mining:** Ensure you are mining on networks and hardware you have permission to use.
-   **Wallet Security:** The system uses a Bitcoin wallet. Ensure your `wallet.dat` or private keys are secured.
-   **Network Traffic:** The miner communicates with the Bitcoin Mainnet and IBM Quantum API. This traffic is visible. Use a VPN (e.g., `kali-anonsurf`) if privacy is a concern.

## Deepseek Integration

The system includes integration with **Deepseek V3.2 Speciale** for advanced agentic reasoning.
By default, it runs in simulation mode. To enable full API access:

1.  Obtain a Deepseek API key.
2.  Export it in your shell:
    ```bash
    export DEEPSEEK_API_KEY="your_key_here"
    ```

## Troubleshooting

-   **TensorFlow Quantum Issues:** Ensure you have the correct CUDA drivers installed if using GPU acceleration.
-   **Memory Errors:** The "Infinite Qubit Extension" can consume significant memory. Adjust `physical_qubits` in `quantum_mining_main.py` if needed.

## Running the Miner

```bash
python3 quantum_mining_main.py
```

