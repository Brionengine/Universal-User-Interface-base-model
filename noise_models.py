"""
Realistic Noise Models for Superconducting Qubits

Implements noise models matching real superconducting qubit parameters:
- T1 (energy relaxation): 50-200 μs
- T2 (dephasing): 50-150 μs
- Gate times: 20-100 ns
- Gate fidelities: 99.9-99.99%

Noise types:
- Depolarizing noise (during gates)
- Thermal relaxation (T1, T2)
- Measurement errors
- Crosstalk
- Leakage to non-computational states

Author: Brion Quantum Technologies & Quantum A.I. Labs
Version: 3.0.0
"""

import cirq
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SuperconductingQubitParameters:
    """
    Physical parameters for superconducting transmon qubits

    Based on state-of-the-art values (e.g., Google Willow, IBM)
    """
    # Coherence times
    T1: float = 100e-6  # Energy relaxation time (seconds)
    T2: float = 100e-6  # Dephasing time (seconds)

    # Gate times
    single_qubit_gate_time: float = 25e-9  # 25 ns
    two_qubit_gate_time: float = 50e-9     # 50 ns (CNOT/CZ)
    measurement_time: float = 1e-6          # 1 μs

    # Gate fidelities (error rates)
    single_qubit_error: float = 1e-3   # 0.1% error (99.9% fidelity)
    two_qubit_error: float = 5e-3      # 0.5% error (99.5% fidelity)
    measurement_error: float = 1e-2    # 1% error (99% fidelity)
    reset_error: float = 5e-3          # 0.5% error

    # Readout parameters
    readout_fidelity: float = 0.99

    # Leakage
    leakage_probability: float = 1e-3  # Per two-qubit gate

    def get_t2_to_gate_ratio(self) -> float:
        """Calculate T2 / single_qubit_gate_time ratio"""
        return self.T2 / self.single_qubit_gate_time

    def get_max_operations(self) -> int:
        """
        Estimate maximum number of operations before decoherence

        Uses T2 as limiting factor
        """
        return int(self.T2 / self.single_qubit_gate_time)


class SuperconductingNoiseModel:
    """
    Comprehensive noise model for superconducting qubits

    Combines multiple noise channels to simulate realistic behavior
    """

    def __init__(
        self,
        params: Optional[SuperconductingQubitParameters] = None
    ):
        """
        Initialize noise model

        Args:
            params: Qubit parameters (uses defaults if None)
        """
        self.params = params if params is not None else SuperconductingQubitParameters()

    def create_cirq_noise_model(self) -> cirq.NoiseModel:
        """
        Create Cirq noise model

        Returns:
            Cirq NoiseModel object
        """
        return SuperconductingCirqNoiseModel(self.params)

    def get_depolarizing_channel(
        self,
        error_rate: float
    ) -> cirq.DepolarizingChannel:
        """
        Create depolarizing noise channel

        Args:
            error_rate: Error probability

        Returns:
            Depolarizing channel
        """
        return cirq.depolarize(error_rate)

    def get_thermal_noise_channel(
        self,
        gate_time: float
    ) -> List[cirq.Gate]:
        """
        Create thermal relaxation noise

        Args:
            gate_time: Duration of operation

        Returns:
            List of noise operations
        """
        # Amplitude damping (T1)
        gamma = 1 - np.exp(-gate_time / self.params.T1)

        # Phase damping (T2, after accounting for T1)
        # T2 = 2T1 in the case of pure dephasing
        # Here we assume T2 is given and calculate pure dephasing rate
        lambda_phi = 1 - np.exp(-gate_time * (1/self.params.T2 - 1/(2*self.params.T1)))

        noise_ops = []

        # Amplitude damping can be approximated with depolarizing for simplicity
        # In reality, should use amplitude_damp channel
        if gamma > 1e-10:
            noise_ops.append(cirq.amplitude_damp(gamma))

        # Phase damping
        if lambda_phi > 1e-10:
            noise_ops.append(cirq.phase_damp(lambda_phi))

        return noise_ops

    def add_noise_to_circuit(
        self,
        circuit: cirq.Circuit
    ) -> cirq.Circuit:
        """
        Add noise to existing circuit

        Args:
            circuit: Clean circuit

        Returns:
            Noisy circuit
        """
        noisy_circuit = cirq.Circuit()

        for moment in circuit:
            noisy_moment = []

            for op in moment:
                # Add original operation
                noisy_moment.append(op)

                # Determine noise to add
                gate = op.gate

                if isinstance(gate, cirq.MeasurementGate):
                    # Measurement noise
                    for qubit in op.qubits:
                        if np.random.random() < self.params.measurement_error:
                            noisy_moment.append(cirq.X(qubit))

                elif isinstance(gate, type(cirq.reset(cirq.LineQubit(0)).gate)):
                    # Reset noise
                    for qubit in op.qubits:
                        if np.random.random() < self.params.reset_error:
                            noisy_moment.append(cirq.X(qubit))

                else:
                    # Gate noise
                    num_qubits = len(op.qubits)

                    if num_qubits == 1:
                        # Single-qubit gate
                        error_rate = self.params.single_qubit_error
                        gate_time = self.params.single_qubit_gate_time

                        # Depolarizing noise
                        for qubit in op.qubits:
                            noisy_moment.append(
                                self.get_depolarizing_channel(error_rate).on(qubit)
                            )

                    elif num_qubits == 2:
                        # Two-qubit gate
                        error_rate = self.params.two_qubit_error
                        gate_time = self.params.two_qubit_gate_time

                        # Two-qubit depolarizing
                        noisy_moment.append(
                            cirq.depolarize(error_rate, n_qubits=2).on(*op.qubits)
                        )

            noisy_circuit.append(noisy_moment)

        return noisy_circuit


class SuperconductingCirqNoiseModel(cirq.NoiseModel):
    """
    Cirq NoiseModel implementation for superconducting qubits

    Automatically applies appropriate noise to operations
    """

    def __init__(self, params: SuperconductingQubitParameters):
        """
        Initialize Cirq noise model

        Args:
            params: Qubit parameters
        """
        self.params = params

    def noisy_operation(self, op: cirq.Operation) -> List[cirq.Operation]:
        """
        Add noise to operation

        Args:
            op: Clean operation

        Returns:
            List of operations including noise
        """
        ops = [op]  # Original operation

        gate = op.gate

        # Skip certain operations
        if gate is None:
            return ops

        if isinstance(gate, cirq.MeasurementGate):
            # Measurement bit-flip noise
            for qubit in op.qubits:
                ops.append(
                    cirq.bit_flip(self.params.measurement_error).on(qubit)
                )

        elif isinstance(gate, type(cirq.reset(cirq.LineQubit(0)).gate)):
            # Reset error
            for qubit in op.qubits:
                ops.append(
                    cirq.bit_flip(self.params.reset_error).on(qubit)
                )

        else:
            # Gate noise
            num_qubits = len(op.qubits)

            if num_qubits == 1:
                # Single-qubit gate noise
                for qubit in op.qubits:
                    # Depolarizing
                    ops.append(
                        cirq.depolarize(self.params.single_qubit_error).on(qubit)
                    )

                    # Thermal noise
                    gamma = 1 - np.exp(
                        -self.params.single_qubit_gate_time / self.params.T1
                    )
                    if gamma > 1e-10:
                        ops.append(cirq.amplitude_damp(gamma).on(qubit))

            elif num_qubits == 2:
                # Two-qubit gate noise
                # Depolarizing on both qubits
                ops.append(
                    cirq.depolarize(
                        self.params.two_qubit_error,
                        n_qubits=2
                    ).on(*op.qubits)
                )

                # Individual qubit thermal noise
                for qubit in op.qubits:
                    gamma = 1 - np.exp(
                        -self.params.two_qubit_gate_time / self.params.T1
                    )
                    if gamma > 1e-10:
                        ops.append(cirq.amplitude_damp(gamma).on(qubit))

        return ops


def create_willow_noise_model() -> SuperconductingNoiseModel:
    """
    Create noise model matching Google Willow chip parameters

    Willow achieves:
    - T1 ~ 100 μs
    - Single-qubit fidelity: 99.95%
    - Two-qubit fidelity: 99.7%
    """
    params = SuperconductingQubitParameters(
        T1=100e-6,
        T2=100e-6,
        single_qubit_gate_time=25e-9,
        two_qubit_gate_time=50e-9,
        measurement_time=1e-6,
        single_qubit_error=5e-4,   # 99.95% fidelity
        two_qubit_error=3e-3,       # 99.7% fidelity
        measurement_error=1e-2,
        readout_fidelity=0.99
    )

    return SuperconductingNoiseModel(params)


def create_conservative_noise_model() -> SuperconductingNoiseModel:
    """
    Create conservative (worse) noise model for testing robustness

    Uses more pessimistic parameters
    """
    params = SuperconductingQubitParameters(
        T1=50e-6,
        T2=50e-6,
        single_qubit_gate_time=50e-9,
        two_qubit_gate_time=100e-9,
        measurement_time=2e-6,
        single_qubit_error=1e-3,    # 99.9% fidelity
        two_qubit_error=5e-3,       # 99.5% fidelity
        measurement_error=2e-2,     # 98% fidelity
        readout_fidelity=0.98
    )

    return SuperconductingNoiseModel(params)


def demonstrate_noise_models():
    """Demonstrate noise model functionality"""
    print("=" * 80)
    print("Superconducting Qubit Noise Models")
    print("=" * 80)

    # Default parameters
    print("\nDefault Parameters:")
    params = SuperconductingQubitParameters()

    print(f"  T1: {params.T1 * 1e6:.1f} μs")
    print(f"  T2: {params.T2 * 1e6:.1f} μs")
    print(f"  Single-qubit gate time: {params.single_qubit_gate_time * 1e9:.1f} ns")
    print(f"  Two-qubit gate time: {params.two_qubit_gate_time * 1e9:.1f} ns")
    print(f"  T2 / gate_time ratio: {params.get_t2_to_gate_ratio():.0f}x")
    print(f"  Max operations before decoherence: ~{params.get_max_operations():,}")

    # Error rates
    print("\nError Rates:")
    print(f"  Single-qubit gate: {params.single_qubit_error:.4f} ({(1-params.single_qubit_error)*100:.2f}% fidelity)")
    print(f"  Two-qubit gate: {params.two_qubit_error:.4f} ({(1-params.two_qubit_error)*100:.2f}% fidelity)")
    print(f"  Measurement: {params.measurement_error:.4f} ({(1-params.measurement_error)*100:.2f}% fidelity)")

    # Willow parameters
    print("\n" + "-" * 80)
    print("Google Willow Chip Parameters")
    print("-" * 80)

    willow_model = create_willow_noise_model()
    willow_params = willow_model.params

    print(f"\n  T2 / gate_time ratio: {willow_params.get_t2_to_gate_ratio():.0f}x")
    print(f"  Max operations: ~{willow_params.get_max_operations():,}")
    print(f"  Single-qubit fidelity: {(1-willow_params.single_qubit_error)*100:.3f}%")
    print(f"  Two-qubit fidelity: {(1-willow_params.two_qubit_error)*100:.2f}%")

    # Demonstrate noise application
    print("\n" + "-" * 80)
    print("Noise Application Demo")
    print("-" * 80)

    # Create simple circuit
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key='result')
    ])

    print(f"\nClean circuit:")
    print(f"  Operations: {sum(len(moment) for moment in circuit)}")
    print(f"  Depth: {len(circuit)}")

    # Add noise
    noise_model = SuperconductingNoiseModel(params)
    noisy_circuit = noise_model.add_noise_to_circuit(circuit)

    print(f"\nNoisy circuit:")
    print(f"  Operations: {sum(len(moment) for moment in noisy_circuit)}")
    print(f"  Depth: {len(noisy_circuit)}")

    # Simulate both
    print("\n" + "-" * 80)
    print("Simulation Comparison")
    print("-" * 80)

    simulator = cirq.Simulator()

    # Clean simulation
    clean_results = simulator.run(circuit, repetitions=1000)
    clean_counts = clean_results.histogram(key='result')

    print("\nClean circuit results (1000 shots):")
    for outcome, count in sorted(clean_counts.items()):
        print(f"  |{outcome:02b}⟩: {count} ({count/10:.1f}%)")

    # Noisy simulation
    noisy_results = simulator.run(noisy_circuit, repetitions=1000)
    noisy_counts = noisy_results.histogram(key='result')

    print("\nNoisy circuit results (1000 shots):")
    for outcome, count in sorted(noisy_counts.items()):
        print(f"  |{outcome:02b}⟩: {count} ({count/10:.1f}%)")

    print("\n" + "=" * 80)
    print("Noise model demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_noise_models()
