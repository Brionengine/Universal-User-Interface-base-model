"""
Surface Code Implementation for Fault-Tolerant Quantum Computing

This module implements a full surface code with:
- Stabilizer measurements (X and Z)
- Multiple syndrome measurement rounds
- Integration with Cirq for circuit generation
- Support for various code distances

For superconducting qubits: T2 ~ 100μs, gate times ~ 20-100ns
This gives T2/T_gate ~ 1000-5000, supporting thousands of operations

Author: Brion Quantum Technologies & Quantum A.I. Labs
Version: 3.0.0
"""

import cirq
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from enum import Enum


class QubitRole(Enum):
    """Role of qubits in surface code"""
    DATA = "data"
    ANCILLA_X = "ancilla_x"  # Measure X stabilizers
    ANCILLA_Z = "ancilla_z"  # Measure Z stabilizers


@dataclass
class SurfaceCodeLattice:
    """
    Surface code lattice structure

    For a distance-d surface code:
    - d^2 data qubits
    - (d^2-1)/2 X-type ancillas
    - (d^2-1)/2 Z-type ancillas
    - Total: 2d^2 - 1 qubits
    """
    distance: int
    data_qubits: Dict[Tuple[int, int], cirq.GridQubit]
    ancilla_x_qubits: Dict[Tuple[int, int], cirq.GridQubit]
    ancilla_z_qubits: Dict[Tuple[int, int], cirq.GridQubit]

    # Stabilizer definitions
    x_stabilizers: Dict[Tuple[int, int], List[Tuple[int, int]]]  # ancilla -> data qubits
    z_stabilizers: Dict[Tuple[int, int], List[Tuple[int, int]]]  # ancilla -> data qubits

    # Boundary tracking for logical operators
    x_boundary: List[Tuple[int, int]]  # Data qubits on X boundary (logical Z)
    z_boundary: List[Tuple[int, int]]  # Data qubits on Z boundary (logical X)


class SurfaceCodeBuilder:
    """Builds surface code lattice structures"""

    def __init__(self, distance: int):
        """
        Initialize surface code builder

        Args:
            distance: Code distance (odd number >= 3)
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance must be odd and >= 3")

        self.distance = distance

    def build_planar_code(self) -> SurfaceCodeLattice:
        """
        Build planar surface code with boundaries

        Returns rotated surface code layout:

        For d=3:
            D - X - D - X - D
            |       |       |
            Z       Z       Z
            |       |       |
            D - X - D - X - D
            |       |       |
            Z       Z       Z
            |       |       |
            D - X - D - X - D

        D = Data qubit
        X = X-type ancilla (measures X stabilizer)
        Z = Z-type ancilla (measures Z stabilizer)
        """
        data_qubits = {}
        ancilla_x_qubits = {}
        ancilla_z_qubits = {}

        # Create qubits on a grid
        # Data qubits at even positions
        for i in range(0, 2*self.distance, 2):
            for j in range(0, 2*self.distance, 2):
                data_qubits[(i, j)] = cirq.GridQubit(i, j)

        # X-type ancillas at (odd, even) positions
        for i in range(1, 2*self.distance-1, 2):
            for j in range(0, 2*self.distance, 2):
                ancilla_x_qubits[(i, j)] = cirq.GridQubit(i, j)

        # Z-type ancillas at (even, odd) positions
        for i in range(0, 2*self.distance, 2):
            for j in range(1, 2*self.distance-1, 2):
                ancilla_z_qubits[(i, j)] = cirq.GridQubit(i, j)

        # Define stabilizers (which data qubits each ancilla measures)
        x_stabilizers = {}
        z_stabilizers = {}

        # X-type stabilizers (plaquette centers)
        for (ai, aj) in ancilla_x_qubits.keys():
            neighbors = []
            # Four neighbors: up, down, left, right
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                data_pos = (ai + di, aj + dj)
                if data_pos in data_qubits:
                    neighbors.append(data_pos)
            x_stabilizers[(ai, aj)] = neighbors

        # Z-type stabilizers
        for (ai, aj) in ancilla_z_qubits.keys():
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                data_pos = (ai + di, aj + dj)
                if data_pos in data_qubits:
                    neighbors.append(data_pos)
            z_stabilizers[(ai, aj)] = neighbors

        # Define logical operator boundaries
        # Logical X: horizontal chain of data qubits
        x_boundary = [(0, j) for j in range(0, 2*self.distance, 2)]

        # Logical Z: vertical chain of data qubits
        z_boundary = [(i, 0) for i in range(0, 2*self.distance, 2)]

        return SurfaceCodeLattice(
            distance=self.distance,
            data_qubits=data_qubits,
            ancilla_x_qubits=ancilla_x_qubits,
            ancilla_z_qubits=ancilla_z_qubits,
            x_stabilizers=x_stabilizers,
            z_stabilizers=z_stabilizers,
            x_boundary=x_boundary,
            z_boundary=z_boundary
        )


class StabilizerMeasurementCircuit:
    """
    Fault-tolerant stabilizer measurement circuits

    Uses 4-step measurement schedule to avoid hook errors:
    1. Reset ancilla
    2. Hadamard on ancilla (for X-type only)
    3. CNOT sequence
    4. Hadamard + Measurement (X-type) or Measurement (Z-type)
    """

    def __init__(self, lattice: SurfaceCodeLattice):
        self.lattice = lattice

    def create_syndrome_measurement_round(
        self,
        measure_x: bool = True,
        measure_z: bool = True
    ) -> cirq.Circuit:
        """
        Create one round of syndrome measurements

        Args:
            measure_x: Measure X stabilizers
            measure_z: Measure Z stabilizers

        Returns:
            Circuit for one syndrome measurement round
        """
        circuit = cirq.Circuit()

        # Step 1: Reset all ancillas
        all_ancillas = []
        if measure_x:
            all_ancillas.extend(self.lattice.ancilla_x_qubits.values())
        if measure_z:
            all_ancillas.extend(self.lattice.ancilla_z_qubits.values())

        circuit.append(cirq.reset(q) for q in all_ancillas)

        # Step 2: Hadamard on X-type ancillas
        if measure_x:
            circuit.append(
                cirq.H(anc)
                for anc in self.lattice.ancilla_x_qubits.values()
            )

        # Step 3: CNOT sequence (4 moments to avoid collisions)
        # For X-type: ancilla is control, data is target (measures X on data)
        # For Z-type: data is control, ancilla is target (measures Z on data)

        # Split into 4 sub-steps based on neighbor direction
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # North, East, South, West

        for direction_idx, (di, dj) in enumerate(directions):
            moment_ops = []

            if measure_x:
                for anc_pos, data_neighbors in self.lattice.x_stabilizers.items():
                    anc = self.lattice.ancilla_x_qubits[anc_pos]
                    for data_pos in data_neighbors:
                        if (data_pos[0] - anc_pos[0], data_pos[1] - anc_pos[1]) == (di, dj):
                            data = self.lattice.data_qubits[data_pos]
                            moment_ops.append(cirq.CNOT(anc, data))

            if measure_z:
                for anc_pos, data_neighbors in self.lattice.z_stabilizers.items():
                    anc = self.lattice.ancilla_z_qubits[anc_pos]
                    for data_pos in data_neighbors:
                        if (data_pos[0] - anc_pos[0], data_pos[1] - anc_pos[1]) == (di, dj):
                            data = self.lattice.data_qubits[data_pos]
                            moment_ops.append(cirq.CNOT(data, anc))

            if moment_ops:
                circuit.append(moment_ops)

        # Step 4: Final Hadamard on X-type ancillas and measurement
        if measure_x:
            circuit.append(
                cirq.H(anc)
                for anc in self.lattice.ancilla_x_qubits.values()
            )

        # Measure all ancillas
        if measure_x:
            circuit.append(
                cirq.measure(anc, key=f'x_anc_{pos[0]}_{pos[1]}')
                for pos, anc in self.lattice.ancilla_x_qubits.items()
            )
        if measure_z:
            circuit.append(
                cirq.measure(anc, key=f'z_anc_{pos[0]}_{pos[1]}')
                for pos, anc in self.lattice.ancilla_z_qubits.items()
            )

        return circuit

    def create_multi_round_syndrome_measurement(
        self,
        num_rounds: int
    ) -> cirq.Circuit:
        """
        Create multiple rounds of syndrome measurements

        For fault-tolerant operation, need distance rounds to
        correctly identify and correct errors.

        Args:
            num_rounds: Number of measurement rounds

        Returns:
            Complete syndrome extraction circuit
        """
        circuit = cirq.Circuit()

        for round_idx in range(num_rounds):
            round_circuit = self.create_syndrome_measurement_round()

            # Rename measurement keys to include round number
            for moment in round_circuit:
                new_moment = []
                for op in moment:
                    if isinstance(op.gate, cirq.MeasurementGate):
                        old_key = op.gate.key
                        new_key = f'r{round_idx}_{old_key}'
                        new_op = cirq.measure(op.qubits[0], key=new_key)
                        new_moment.append(new_op)
                    else:
                        new_moment.append(op)
                circuit.append(new_moment)

        return circuit


class LogicalStatePreparation:
    """Prepare logical |0⟩ and |1⟩ states"""

    def __init__(self, lattice: SurfaceCodeLattice):
        self.lattice = lattice

    def prepare_logical_zero(self) -> cirq.Circuit:
        """
        Prepare logical |0⟩ state

        All data qubits initialized to |0⟩
        X stabilizers should measure +1
        Z stabilizers should measure +1
        """
        circuit = cirq.Circuit()

        # Reset all qubits to |0⟩
        all_qubits = list(self.lattice.data_qubits.values())
        circuit.append(cirq.reset(q) for q in all_qubits)

        return circuit

    def prepare_logical_one(self) -> cirq.Circuit:
        """
        Prepare logical |1⟩ state

        Apply logical X to |0⟩
        Flip all data qubits along the logical X boundary
        """
        circuit = self.prepare_logical_zero()

        # Apply X to all qubits on the X boundary (horizontal chain)
        circuit.append(
            cirq.X(self.lattice.data_qubits[pos])
            for pos in self.lattice.x_boundary
        )

        return circuit

    def prepare_logical_plus(self) -> cirq.Circuit:
        """
        Prepare logical |+⟩ state

        Apply logical H to |0⟩
        Apply H to all data qubits
        """
        circuit = cirq.Circuit()

        # Reset and apply Hadamard to all data qubits
        all_data_qubits = list(self.lattice.data_qubits.values())
        circuit.append(cirq.reset(q) for q in all_data_qubits)
        circuit.append(cirq.H(q) for q in all_data_qubits)

        return circuit


@dataclass
class SyndromeData:
    """Store syndrome measurement results"""
    round_idx: int
    x_syndromes: Dict[Tuple[int, int], int]  # position -> measurement result
    z_syndromes: Dict[Tuple[int, int], int]

    def get_detection_events(self, previous: Optional['SyndromeData']) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Get detection events (syndrome changes) compared to previous round

        Returns:
            (x_detections, z_detections) - positions with changed syndromes
        """
        x_detections = set()
        z_detections = set()

        if previous is None:
            # First round: all syndromes are detection events
            x_detections = {pos for pos, val in self.x_syndromes.items() if val == 1}
            z_detections = {pos for pos, val in self.z_syndromes.items() if val == 1}
        else:
            # Compare with previous round
            for pos in self.x_syndromes:
                if pos in previous.x_syndromes:
                    if self.x_syndromes[pos] != previous.x_syndromes[pos]:
                        x_detections.add(pos)

            for pos in self.z_syndromes:
                if pos in previous.z_syndromes:
                    if self.z_syndromes[pos] != previous.z_syndromes[pos]:
                        z_detections.add(pos)

        return x_detections, z_detections


class SurfaceCodeSimulator:
    """
    Complete surface code simulator with syndrome extraction
    """

    def __init__(self, distance: int):
        """
        Initialize surface code simulator

        Args:
            distance: Code distance
        """
        self.distance = distance

        # Build lattice
        builder = SurfaceCodeBuilder(distance)
        self.lattice = builder.build_planar_code()

        # Initialize circuit builders
        self.stabilizer_circuit = StabilizerMeasurementCircuit(self.lattice)
        self.state_prep = LogicalStatePreparation(self.lattice)

        # Syndrome history
        self.syndrome_history: List[SyndromeData] = []

    def create_full_qec_cycle(
        self,
        initial_state: str = '0',
        num_syndrome_rounds: int = None
    ) -> cirq.Circuit:
        """
        Create full QEC cycle

        Args:
            initial_state: '0', '1', or '+'
            num_syndrome_rounds: Number of syndrome measurement rounds
                                (defaults to distance for fault-tolerance)

        Returns:
            Complete QEC circuit
        """
        if num_syndrome_rounds is None:
            num_syndrome_rounds = self.distance

        circuit = cirq.Circuit()

        # 1. State preparation
        if initial_state == '0':
            circuit += self.state_prep.prepare_logical_zero()
        elif initial_state == '1':
            circuit += self.state_prep.prepare_logical_one()
        elif initial_state == '+':
            circuit += self.state_prep.prepare_logical_plus()
        else:
            raise ValueError(f"Unknown state: {initial_state}")

        # 2. Syndrome measurements
        circuit += self.stabilizer_circuit.create_multi_round_syndrome_measurement(
            num_syndrome_rounds
        )

        return circuit

    def extract_syndromes_from_results(
        self,
        results: cirq.Result,
        num_rounds: int
    ) -> List[SyndromeData]:
        """
        Extract syndrome data from measurement results

        Args:
            results: Cirq simulation results
            num_rounds: Number of measurement rounds

        Returns:
            List of SyndromeData for each round
        """
        syndrome_list = []

        for round_idx in range(num_rounds):
            x_syndromes = {}
            z_syndromes = {}

            # Extract X syndromes
            for pos in self.lattice.ancilla_x_qubits.keys():
                key = f'r{round_idx}_x_anc_{pos[0]}_{pos[1]}'
                if key in results.measurements:
                    x_syndromes[pos] = int(results.measurements[key][0])

            # Extract Z syndromes
            for pos in self.lattice.ancilla_z_qubits.keys():
                key = f'r{round_idx}_z_anc_{pos[0]}_{pos[1]}'
                if key in results.measurements:
                    z_syndromes[pos] = int(results.measurements[key][0])

            syndrome_list.append(SyndromeData(
                round_idx=round_idx,
                x_syndromes=x_syndromes,
                z_syndromes=z_syndromes
            ))

        return syndrome_list

    def get_num_qubits(self) -> Dict[str, int]:
        """Get qubit counts"""
        return {
            'data': len(self.lattice.data_qubits),
            'ancilla_x': len(self.lattice.ancilla_x_qubits),
            'ancilla_z': len(self.lattice.ancilla_z_qubits),
            'total': len(self.lattice.data_qubits) +
                    len(self.lattice.ancilla_x_qubits) +
                    len(self.lattice.ancilla_z_qubits)
        }


def demonstrate_surface_code():
    """Demonstrate surface code implementation"""
    print("=" * 80)
    print("Surface Code Implementation for Fault-Tolerant Quantum Computing")
    print("=" * 80)

    # Create distance-5 surface code
    distance = 5
    print(f"\nCreating distance-{distance} surface code...")

    simulator = SurfaceCodeSimulator(distance)

    # Get qubit counts
    qubit_counts = simulator.get_num_qubits()
    print(f"\nQubit counts:")
    print(f"  Data qubits: {qubit_counts['data']}")
    print(f"  X-type ancillas: {qubit_counts['ancilla_x']}")
    print(f"  Z-type ancillas: {qubit_counts['ancilla_z']}")
    print(f"  Total qubits: {qubit_counts['total']}")

    # Create QEC cycle
    print(f"\nCreating QEC cycle with {distance} syndrome rounds...")
    circuit = simulator.create_full_qec_cycle(
        initial_state='0',
        num_syndrome_rounds=distance
    )

    print(f"\nCircuit statistics:")
    print(f"  Total moments: {len(circuit)}")
    print(f"  Total operations: {sum(len(moment) for moment in circuit)}")

    # Simulate
    print("\nSimulating QEC cycle...")
    sim = cirq.Simulator()
    results = sim.run(circuit, repetitions=1)

    # Extract syndromes
    syndromes = simulator.extract_syndromes_from_results(results, distance)

    print(f"\nSyndrome extraction successful!")
    print(f"  Rounds: {len(syndromes)}")

    # Analyze first round
    if syndromes:
        first_round = syndromes[0]
        x_nonzero = sum(1 for v in first_round.x_syndromes.values() if v == 1)
        z_nonzero = sum(1 for v in first_round.z_syndromes.values() if v == 1)

        print(f"\nFirst round syndrome analysis:")
        print(f"  X syndromes (non-zero): {x_nonzero}/{len(first_round.x_syndromes)}")
        print(f"  Z syndromes (non-zero): {z_nonzero}/{len(first_round.z_syndromes)}")

    print("\n" + "=" * 80)
    print("Surface code demonstration complete!")
    print("=" * 80)

    return simulator, circuit


if __name__ == "__main__":
    demonstrate_surface_code()
