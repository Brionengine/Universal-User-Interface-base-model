# Universal User Interface (UUI) Base Model

This repository contains the base Python and JavaScript assets for the Universal User Interface project, including quantum/ML integration helpers, automation utilities, and a lightweight Node.js database connector.

## Repository layout
- `UUI.py`: Core UI model with layout, component, theme, and session helpers.
- `UUIAutomation.py`: Task scheduler/automation helper for orchestrating UI workflows.
- `deepseek_integration.py`: Deepseek V3.2 Speciale agent for quantum-state analysis and mining strategy guidance.
- `mlperf_nvidia_integration.py`: Nvidia MLPerf v5.1 agent for benchmark simulation and optimization routines.
- `benchmark_cluster.py`, `test_infinite_qubit.py`: Benchmark and test harnesses for large-scale or infinite-qubit simulations.
- `DBConnector/`: Node.js connector (see below) with environment config and package list.
- Encryption/crypto helpers: `CoreEncryptionEngine.js`, `High Decryption Client-Side Example.js`, `Key Rotation Support.js`.
- Utility guides: `Setting Up the Environment.txt`, `File Structure.txt`, `Running the Server.js`, `Server Configuration.js`, `Toggle.py`, `Toggle Ext.txt`.
- Platform setup scripts: `setup_kali_env.sh`, `install_tf_quantum.sh`, plus `KALI_LINUX_SETUP.md` guidance.

## Python quick start
```bash
python UUI.py          # run the base UUI example
python UUIAutomation.py # run the task scheduler demo
```

### Using the integration agents
```python
from Universal_User_Interface_base_model import DeepseekSpecialeAgent, NvidiaMLPerfAgent

deepseek = DeepseekSpecialeAgent()
state_summary = deepseek.analyze_quantum_state({"qubits": 64, "gates": 120})

mlperf = NvidiaMLPerfAgent()
benchmark = mlperf.run_benchmark("hybrid-quantum-sim")
```

## Node.js database connector
The `DBConnector` folder contains a minimal Express-based connector.

```bash
cd DBConnector
npm install --save $(cat Packages.txt)  # install listed packages
node server.js                          # start the connector
```

Environment variables can be provided via `Config.env`; `Server/server.js` contains the app entry if you prefer the nested structure noted in `File Structure.txt`.

## Kali Linux setup
For a Kali-focused setup (dependency installation and verification), run:

```bash
chmod +x setup_kali_env.sh
./setup_kali_env.sh
```

See [KALI_LINUX_SETUP.md](KALI_LINUX_SETUP.md) for detailed guidance and validation steps.
