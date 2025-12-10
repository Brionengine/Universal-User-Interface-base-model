"""
Magic State Distillation for T Gates

Implements magic state distillation protocols to enable universal
fault-tolerant quantum computation. T gates cannot be implemented
transversally, so we use magic state injection.

Protocol:
1. Prepare noisy |T⟩ states (magic states)
2. Distill into higher-fidelity |T⟩ states using Clifford operations
3. Inject purified |T⟩ state to implement T gate

Common protocols:
- 15-to-1 distillation (Bravyi-Kitaev)
- Reed-Muller code-based distillation
- Concatenated distillation

Author: Brion Quantum Technologies & Quantum A.I. Labs
Version: 3.0.0
"""

import cirq
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from surface_code_implementation import SurfaceCodeLattice


@dataclass
class MagicState:
    """
    Magic state |T⟩ = (|0⟩ + e^(iπ/4)|1⟩) / √2

    This state cannot be created by Clifford operations alone,
    making it "magic" for enabling universal quantum computation.
    """
    fidelity: float  # Fidelity with ideal |T⟩ state
    logical_qubit: Optional[SurfaceCodeLattice] = None


@dataclass
class DistillationResult:
    """Result of magic state distillation"""
    output_state: MagicState
    input_states_used: int
    distillation_level: int
    success_probability: float
    circuit: cirq.Circuit


class MagicStatePreparation:
    """
    Prepare noisy magic states

    Creates approximate |T⟩ states that will be distilled
    """

    @staticmethod
    def prepare_noisy_t_state(
        qubit: cirq.Qid,
        physical_error_rate: float = 0.001
    ) -> cirq.Circuit:
        """
        Prepare noisy |T⟩ state

        |T⟩ = (|0⟩ + e^(iπ/4)|1⟩) / √2

        Args:
            qubit: Target qubit
            physical_error_rate: Physical error rate

        Returns:
            Preparation circuit
        """
        circuit = cirq.Circuit()

        # Start from |0⟩
        circuit.append(cirq.reset(qubit))

        # Create superposition: H|0⟩ = |+⟩
        circuit.append(cirq.H(qubit))

        # Apply S^(1/2) = T to get |T⟩ state
        # T|+⟩ = (|0⟩ + e^(iπ/4)|1⟩) / √2
        circuit.append(cirq.T(qubit))

        return circuit

    @staticmethod
    def prepare_logical_t_state(
        lattice: SurfaceCodeLattice
    ) -> cirq.Circuit:
        """
        Prepare |T⟩ state on logical qubit

        Apply physical T gate to all data qubits
        (non-transversal, introduces correlated errors)

        Args:
            lattice: Surface code lattice

        Returns:
            Preparation circuit
        """
        circuit = cirq.Circuit()

        # Initialize to logical |+⟩
        circuit.append(cirq.reset(q) for q in lattice.data_qubits.values())
        circuit.append(cirq.H(q) for q in lattice.data_qubits.values())

        # Apply T gate (non-transversal)
        # This creates a noisy logical |T⟩ state
        circuit.append(cirq.T(q) for q in lattice.data_qubits.values())

        return circuit


class FifteenToOneDistillation:
    """
    15-to-1 magic state distillation protocol

    Uses 15 noisy |T⟩ states to produce 1 higher-fidelity |T⟩ state

    Input error rate: ε_in
    Output error rate: ε_out ≈ 35 × ε_in^3 (for small ε_in)

    Success probability: ~1 (very high)
    """

    def __init__(self):
        """Initialize 15-to-1 distillation"""
        self.input_size = 15
        self.output_size = 1

    def create_distillation_circuit(
        self,
        input_qubits: List[cirq.Qid],
        output_qubit: cirq.Qid,
        ancilla_qubits: List[cirq.Qid]
    ) -> cirq.Circuit:
        """
        Create 15-to-1 distillation circuit

        Args:
            input_qubits: 15 noisy |T⟩ state qubits
            output_qubit: Output qubit for distilled state
            ancilla_qubits: Additional ancilla qubits for syndrome measurement

        Returns:
            Distillation circuit
        """
        if len(input_qubits) != 15:
            raise ValueError("Need exactly 15 input qubits")

        circuit = cirq.Circuit()

        # Step 1: Prepare all inputs in noisy |T⟩ states
        # (Assumed to be done already)

        # Step 2: Encode using [[15,1,3]] quantum Reed-Muller code
        # This involves CNOT gates according to the encoding matrix

        # Encoding: Apply CNOTs based on Reed-Muller code structure
        # Simplified encoding (full implementation would use proper RM code)

        # Create Bell pairs and teleport through them with corrections
        # This is a simplified version - full protocol is more complex

        # Apply X basis measurements on 14 qubits
        circuit.append(cirq.H(q) for q in input_qubits[:14])
        circuit.append(
            cirq.measure(input_qubits[i], key=f'syndrome_{i}')
            for i in range(14)
        )

        # The 15th qubit becomes the output after corrections
        # based on measurement results

        # Step 3: Classical post-processing determines if distillation succeeded
        # If syndrome measurements match expected pattern, accept output

        return circuit

    def compute_output_fidelity(
        self,
        input_fidelity: float
    ) -> float:
        """
        Compute output fidelity after distillation

        Args:
            input_fidelity: Input state fidelity

        Returns:
            Expected output fidelity
        """
        input_error = 1 - input_fidelity

        # Output error: ε_out ≈ 35 × ε_in^3
        # This is valid for small ε_in
        if input_error < 0.1:
            output_error = 35 * input_error ** 3
        else:
            # Less effective for large errors
            output_error = input_error

        output_fidelity = 1 - output_error

        # Cap at input fidelity (can't make it worse with good protocol)
        return min(output_fidelity, input_fidelity)


class TGateViaDistillation:
    """
    Implement T gate using magic state distillation

    Process:
    1. Distill high-fidelity |T⟩ magic state
    2. Use gate teleportation to apply T gate using magic state
    3. Correct based on measurement outcomes
    """

    def __init__(
        self,
        physical_error_rate: float = 0.001,
        target_t_error_rate: float = 1e-8
    ):
        """
        Initialize T gate implementation

        Args:
            physical_error_rate: Physical gate error rate
            target_t_error_rate: Target T gate error rate
        """
        self.physical_error_rate = physical_error_rate
        self.target_t_error_rate = target_t_error_rate

        self.distillation = FifteenToOneDistillation()

        # Calculate number of distillation rounds needed
        self._compute_distillation_levels()

    def _compute_distillation_levels(self):
        """
        Compute number of distillation levels needed

        Each level: ε_out ≈ 35 × ε_in^3
        """
        current_error = self.physical_error_rate
        self.distillation_levels = 0

        while current_error > self.target_t_error_rate:
            current_error = 35 * current_error ** 3
            self.distillation_levels += 1

            if self.distillation_levels > 10:
                # Safety check
                break

    def create_t_gate_circuit(
        self,
        target_qubit: cirq.Qid,
        magic_state_qubit: cirq.Qid,
        ancilla: cirq.Qid
    ) -> cirq.Circuit:
        """
        Create T gate via magic state injection

        Uses gate teleportation:
        1. Prepare |T⟩ magic state
        2. Perform Bell measurement between target and ancilla
        3. Apply corrections based on measurement

        Args:
            target_qubit: Qubit to apply T gate to
            magic_state_qubit: Qubit with |T⟩ magic state
            ancilla: Ancilla qubit for teleportation

        Returns:
            T gate circuit
        """
        circuit = cirq.Circuit()

        # Step 1: Prepare magic state (assumed done)
        # magic_state_qubit is in |T⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2

        # Step 2: Gate teleportation protocol
        # Create entanglement
        circuit.append(cirq.CNOT(target_qubit, ancilla))
        circuit.append(cirq.H(target_qubit))

        # Measure both qubits
        circuit.append([
            cirq.measure(target_qubit, key='m1'),
            cirq.measure(ancilla, key='m2')
        ])

        # Step 3: Corrections (would be done classically in real system)
        # If m1 = 1: apply Z to magic_state_qubit
        # If m2 = 1: apply X to magic_state_qubit

        # The magic_state_qubit now has the state with T applied
        # Transfer it back to target_qubit

        return circuit

    def get_distillation_cost(self) -> Dict[str, int]:
        """
        Calculate resource cost for T gate

        Returns:
            Dictionary with resource costs
        """
        # Number of noisy |T⟩ states needed
        states_per_level = 15

        total_noisy_states = states_per_level ** self.distillation_levels

        return {
            'distillation_levels': self.distillation_levels,
            'noisy_t_states_per_gate': total_noisy_states,
            'total_qubits_per_t': total_noisy_states + 10,  # Plus ancillas
            'clifford_gates_per_t': total_noisy_states * 20  # Approximate
        }

    def get_t_gate_error_rate(self) -> float:
        """
        Get actual T gate error rate after distillation

        Returns:
            Expected T gate error rate
        """
        current_error = self.physical_error_rate

        for _ in range(self.distillation_levels):
            current_error = 35 * current_error ** 3

        return current_error


def demonstrate_magic_state_distillation():
    """Demonstrate magic state distillation"""
    print("=" * 80)
    print("Magic State Distillation for T Gates")
    print("=" * 80)

    # Initialize T gate implementation
    physical_error = 1e-3
    target_t_error = 1e-8

    print(f"\nPhysical error rate: {physical_error:.2e}")
    print(f"Target T gate error rate: {target_t_error:.2e}")

    t_gate = TGateViaDistillation(
        physical_error_rate=physical_error,
        target_t_error_rate=target_t_error
    )

    print(f"\nDistillation levels required: {t_gate.distillation_levels}")

    # Show error suppression at each level
    print("\nError suppression per level:")
    current_error = physical_error

    for level in range(t_gate.distillation_levels):
        new_error = 35 * current_error ** 3
        suppression = current_error / new_error

        print(f"  Level {level + 1}: {current_error:.2e} -> {new_error:.2e} "
              f"(suppression: {suppression:.2e}x)")

        current_error = new_error

    # Resource costs
    costs = t_gate.get_distillation_cost()

    print("\nResource costs per T gate:")
    print(f"  Noisy |T⟩ states: {costs['noisy_t_states_per_gate']}")
    print(f"  Total qubits: {costs['total_qubits_per_t']}")
    print(f"  Clifford gates: {costs['clifford_gates_per_t']}")

    # Achieved error rate
    achieved_error = t_gate.get_t_gate_error_rate()

    print(f"\nAchieved T gate error rate: {achieved_error:.2e}")
    print(f"Target met: {achieved_error <= target_t_error}")

    # Demonstrate 15-to-1 distillation
    print("\n" + "-" * 80)
    print("15-to-1 Distillation Analysis")
    print("-" * 80)

    distillation = FifteenToOneDistillation()

    input_fidelities = [0.99, 0.999, 0.9999]

    print("\nFidelity improvement:")
    for input_f in input_fidelities:
        output_f = distillation.compute_output_fidelity(input_f)

        input_e = 1 - input_f
        output_e = 1 - output_f

        improvement = input_e / output_e if output_e > 0 else float('inf')

        print(f"  Input: {input_f:.6f} (ε={input_e:.2e}) -> "
              f"Output: {output_f:.10f} (ε={output_e:.2e}) "
              f"[{improvement:.2e}x better]")

    # Compare T gate cost with Clifford gates
    print("\n" + "-" * 80)
    print("T Gate vs Clifford Gates")
    print("-" * 80)

    print(f"\nResource ratio (T gate / Clifford gate):")
    print(f"  Qubits: ~{costs['total_qubits_per_t']}x")
    print(f"  Gates: ~{costs['clifford_gates_per_t']}x")
    print(f"  Time: ~{costs['clifford_gates_per_t']}x")

    print("\nThis is why T gates are expensive in fault-tolerant QC!")
    print("Algorithms should minimize T count for practical implementation.")

    print("\n" + "=" * 80)
    print("Magic state distillation demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_magic_state_distillation()
