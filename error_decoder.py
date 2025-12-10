"""
Error Syndrome Decoder for Surface Codes

Implements Minimum Weight Perfect Matching (MWPM) decoder
for decoding error syndromes in surface codes.

The decoder:
1. Takes syndrome measurements as input
2. Constructs a decoding graph with weighted edges
3. Finds minimum weight perfect matching
4. Determines correction operations

Author: Brion Quantum Technologies & Quantum A.I. Labs
Version: 3.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import networkx as nx
from surface_code_implementation import SurfaceCodeLattice, SyndromeData


@dataclass
class ErrorChain:
    """Represents a chain of errors (correction path)"""
    error_type: str  # 'X' or 'Z'
    data_qubit_positions: List[Tuple[int, int]]
    weight: float


class MinimumWeightPerfectMatching:
    """
    Minimum Weight Perfect Matching decoder

    Uses NetworkX blossom algorithm for efficient matching.
    """

    def __init__(self, lattice: SurfaceCodeLattice):
        """
        Initialize MWPM decoder

        Args:
            lattice: Surface code lattice structure
        """
        self.lattice = lattice

        # Pre-compute distances between all pairs of ancillas
        self._precompute_distances()

    def _precompute_distances(self):
        """
        Precompute Manhattan distances between ancilla pairs

        For a distance-d code, the weight between two syndrome
        locations is their Manhattan distance on the lattice.
        """
        # X-type ancilla distances
        x_positions = list(self.lattice.ancilla_x_qubits.keys())
        self.x_distances = {}

        for i, pos1 in enumerate(x_positions):
            for pos2 in x_positions[i:]:
                dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                self.x_distances[(pos1, pos2)] = dist
                self.x_distances[(pos2, pos1)] = dist

        # Z-type ancilla distances
        z_positions = list(self.lattice.ancilla_z_qubits.keys())
        self.z_distances = {}

        for i, pos1 in enumerate(z_positions):
            for pos2 in z_positions[i:]:
                dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                self.z_distances[(pos1, pos2)] = dist
                self.z_distances[(pos2, pos1)] = dist

    def decode_x_errors(
        self,
        detection_events: Set[Tuple[int, int]]
    ) -> ErrorChain:
        """
        Decode X errors from Z-syndrome detection events

        Z stabilizers detect X errors on data qubits.

        Args:
            detection_events: Set of Z-ancilla positions with detection events

        Returns:
            ErrorChain of X corrections
        """
        if not detection_events:
            return ErrorChain(error_type='X', data_qubit_positions=[], weight=0.0)

        # Convert to list for indexing
        events_list = list(detection_events)

        # Special case: single detection event (error on boundary)
        if len(events_list) == 1:
            # Correct to nearest boundary
            pos = events_list[0]
            correction_path = self._path_to_z_boundary(pos)
            return ErrorChain(
                error_type='X',
                data_qubit_positions=correction_path,
                weight=len(correction_path)
            )

        # Build matching graph
        G = nx.Graph()

        # Add nodes
        for idx, pos in enumerate(events_list):
            G.add_node(idx, position=pos)

        # Add virtual boundary nodes for odd number of detections
        num_boundary_nodes = 0
        if len(events_list) % 2 == 1:
            # Add virtual boundary node
            boundary_idx = len(events_list)
            G.add_node(boundary_idx, position=None, is_boundary=True)
            num_boundary_nodes = 1

        # Add edges with weights (negative for min-weight matching)
        for i in range(len(events_list)):
            for j in range(i + 1, len(events_list)):
                pos1 = events_list[i]
                pos2 = events_list[j]
                weight = self.z_distances.get((pos1, pos2), float('inf'))
                G.add_edge(i, j, weight=-weight)

            # Add edges to boundary if needed
            if num_boundary_nodes > 0:
                pos = events_list[i]
                boundary_dist = min(
                    abs(pos[0]),  # Distance to left boundary
                    abs(pos[1]),  # Distance to top boundary
                    abs(pos[0] - 2*self.lattice.distance),  # Right boundary
                    abs(pos[1] - 2*self.lattice.distance)   # Bottom boundary
                )
                G.add_edge(i, boundary_idx, weight=-boundary_dist)

        # Find maximum weight matching (minimum weight with negative weights)
        matching = nx.max_weight_matching(G, maxcardinality=True)

        # Extract correction paths
        all_corrections = []

        for (idx1, idx2) in matching:
            # Skip boundary pairs
            if idx1 >= len(events_list) or idx2 >= len(events_list):
                # One endpoint is boundary - correct to boundary
                real_idx = idx1 if idx1 < len(events_list) else idx2
                pos = events_list[real_idx]
                path = self._path_to_z_boundary(pos)
                all_corrections.extend(path)
            else:
                # Both endpoints are real detections
                pos1 = events_list[idx1]
                pos2 = events_list[idx2]
                path = self._path_between_z_ancillas(pos1, pos2)
                all_corrections.extend(path)

        # Remove duplicates (even number of corrections cancel out)
        correction_counts = {}
        for pos in all_corrections:
            correction_counts[pos] = correction_counts.get(pos, 0) + 1

        final_corrections = [
            pos for pos, count in correction_counts.items()
            if count % 2 == 1
        ]

        return ErrorChain(
            error_type='X',
            data_qubit_positions=final_corrections,
            weight=len(final_corrections)
        )

    def decode_z_errors(
        self,
        detection_events: Set[Tuple[int, int]]
    ) -> ErrorChain:
        """
        Decode Z errors from X-syndrome detection events

        X stabilizers detect Z errors on data qubits.

        Args:
            detection_events: Set of X-ancilla positions with detection events

        Returns:
            ErrorChain of Z corrections
        """
        if not detection_events:
            return ErrorChain(error_type='Z', data_qubit_positions=[], weight=0.0)

        events_list = list(detection_events)

        # Special case: single detection
        if len(events_list) == 1:
            pos = events_list[0]
            correction_path = self._path_to_x_boundary(pos)
            return ErrorChain(
                error_type='Z',
                data_qubit_positions=correction_path,
                weight=len(correction_path)
            )

        # Build matching graph (similar to X errors)
        G = nx.Graph()

        for idx, pos in enumerate(events_list):
            G.add_node(idx, position=pos)

        num_boundary_nodes = 0
        if len(events_list) % 2 == 1:
            boundary_idx = len(events_list)
            G.add_node(boundary_idx, position=None, is_boundary=True)
            num_boundary_nodes = 1

        for i in range(len(events_list)):
            for j in range(i + 1, len(events_list)):
                pos1 = events_list[i]
                pos2 = events_list[j]
                weight = self.x_distances.get((pos1, pos2), float('inf'))
                G.add_edge(i, j, weight=-weight)

            if num_boundary_nodes > 0:
                pos = events_list[i]
                boundary_dist = min(
                    abs(pos[0]),
                    abs(pos[1]),
                    abs(pos[0] - 2*self.lattice.distance),
                    abs(pos[1] - 2*self.lattice.distance)
                )
                G.add_edge(i, boundary_idx, weight=-boundary_dist)

        matching = nx.max_weight_matching(G, maxcardinality=True)

        all_corrections = []

        for (idx1, idx2) in matching:
            if idx1 >= len(events_list) or idx2 >= len(events_list):
                real_idx = idx1 if idx1 < len(events_list) else idx2
                pos = events_list[real_idx]
                path = self._path_to_x_boundary(pos)
                all_corrections.extend(path)
            else:
                pos1 = events_list[idx1]
                pos2 = events_list[idx2]
                path = self._path_between_x_ancillas(pos1, pos2)
                all_corrections.extend(path)

        # Remove duplicates
        correction_counts = {}
        for pos in all_corrections:
            correction_counts[pos] = correction_counts.get(pos, 0) + 1

        final_corrections = [
            pos for pos, count in correction_counts.items()
            if count % 2 == 1
        ]

        return ErrorChain(
            error_type='Z',
            data_qubit_positions=final_corrections,
            weight=len(final_corrections)
        )

    def _path_between_z_ancillas(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Find path of data qubits between two Z-ancilla positions

        Returns list of data qubit positions to flip
        """
        # Use Manhattan path
        path = []
        current = list(pos1)

        # Move horizontally first
        while current[0] != pos2[0]:
            step = 1 if pos2[0] > current[0] else -1
            current[0] += step
            if current[0] % 2 == 0 and current[1] % 2 == 0:  # Data qubit position
                if tuple(current) in self.lattice.data_qubits:
                    path.append(tuple(current))

        # Then move vertically
        while current[1] != pos2[1]:
            step = 1 if pos2[1] > current[1] else -1
            current[1] += step
            if current[0] % 2 == 0 and current[1] % 2 == 0:  # Data qubit position
                if tuple(current) in self.lattice.data_qubits:
                    path.append(tuple(current))

        return path

    def _path_between_x_ancillas(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Find path of data qubits between two X-ancilla positions"""
        path = []
        current = list(pos1)

        # Move horizontally
        while current[0] != pos2[0]:
            step = 1 if pos2[0] > current[0] else -1
            current[0] += step
            if current[0] % 2 == 0 and current[1] % 2 == 0:
                if tuple(current) in self.lattice.data_qubits:
                    path.append(tuple(current))

        # Move vertically
        while current[1] != pos2[1]:
            step = 1 if pos2[1] > current[1] else -1
            current[1] += step
            if current[0] % 2 == 0 and current[1] % 2 == 0:
                if tuple(current) in self.lattice.data_qubits:
                    path.append(tuple(current))

        return path

    def _path_to_z_boundary(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find shortest path from Z-ancilla position to nearest Z boundary"""
        # Z boundary is vertical (i changes, j=0)
        path = []
        current = list(pos)

        # Move to left boundary (j=0)
        while current[1] > 0:
            current[1] -= 1
            if current[0] % 2 == 0 and current[1] % 2 == 0:
                if tuple(current) in self.lattice.data_qubits:
                    path.append(tuple(current))

        return path

    def _path_to_x_boundary(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find shortest path from X-ancilla position to nearest X boundary"""
        # X boundary is horizontal (i=0, j changes)
        path = []
        current = list(pos)

        # Move to top boundary (i=0)
        while current[0] > 0:
            current[0] -= 1
            if current[0] % 2 == 0 and current[1] % 2 == 0:
                if tuple(current) in self.lattice.data_qubits:
                    path.append(tuple(current))

        return path


class SurfaceCodeDecoder:
    """
    High-level decoder for surface codes

    Manages syndrome history and decoding across multiple rounds
    """

    def __init__(self, lattice: SurfaceCodeLattice):
        """
        Initialize decoder

        Args:
            lattice: Surface code lattice
        """
        self.lattice = lattice
        self.mwpm = MinimumWeightPerfectMatching(lattice)

        # Decoding statistics
        self.stats = {
            'total_rounds_decoded': 0,
            'total_x_corrections': 0,
            'total_z_corrections': 0,
            'avg_correction_weight': 0.0
        }

    def decode_syndrome_history(
        self,
        syndrome_history: List[SyndromeData]
    ) -> Tuple[ErrorChain, ErrorChain]:
        """
        Decode error chains from full syndrome history

        Args:
            syndrome_history: List of syndrome measurements over time

        Returns:
            (x_chain, z_chain) - Error chains for X and Z corrections
        """
        if not syndrome_history:
            return (
                ErrorChain('X', [], 0.0),
                ErrorChain('Z', [], 0.0)
            )

        # Accumulate detection events across all rounds
        x_detection_events = set()
        z_detection_events = set()

        for i, syndrome in enumerate(syndrome_history):
            prev_syndrome = syndrome_history[i-1] if i > 0 else None
            x_det, z_det = syndrome.get_detection_events(prev_syndrome)

            x_detection_events.update(x_det)
            z_detection_events.update(z_det)

        # Decode accumulated detection events
        x_chain = self.mwpm.decode_x_errors(z_detection_events)
        z_chain = self.mwpm.decode_z_errors(x_detection_events)

        # Update statistics
        self.stats['total_rounds_decoded'] += len(syndrome_history)
        self.stats['total_x_corrections'] += len(x_chain.data_qubit_positions)
        self.stats['total_z_corrections'] += len(z_chain.data_qubit_positions)

        total_corrections = (
            self.stats['total_x_corrections'] +
            self.stats['total_z_corrections']
        )
        if self.stats['total_rounds_decoded'] > 0:
            self.stats['avg_correction_weight'] = (
                total_corrections / self.stats['total_rounds_decoded']
            )

        return x_chain, z_chain

    def apply_corrections(
        self,
        x_chain: ErrorChain,
        z_chain: ErrorChain
    ) -> Dict[Tuple[int, int], List[str]]:
        """
        Generate correction operations

        Args:
            x_chain: X error chain
            z_chain: Z error chain

        Returns:
            Dictionary mapping data qubit positions to correction gates
        """
        corrections = {}

        # X corrections (flip X errors)
        for pos in x_chain.data_qubit_positions:
            if pos not in corrections:
                corrections[pos] = []
            corrections[pos].append('X')

        # Z corrections (flip Z errors)
        for pos in z_chain.data_qubit_positions:
            if pos not in corrections:
                corrections[pos] = []
            corrections[pos].append('Z')

        return corrections

    def get_decoding_statistics(self) -> Dict:
        """Get decoding statistics"""
        return self.stats.copy()


def demonstrate_decoder():
    """Demonstrate error decoder"""
    from surface_code_implementation import SurfaceCodeBuilder

    print("=" * 80)
    print("Surface Code Error Decoder Demonstration")
    print("=" * 80)

    # Create distance-5 surface code
    distance = 5
    builder = SurfaceCodeBuilder(distance)
    lattice = builder.build_planar_code()

    print(f"\nCreated distance-{distance} surface code")

    # Initialize decoder
    decoder = SurfaceCodeDecoder(lattice)
    print("Decoder initialized with MWPM algorithm")

    # Simulate some detection events
    print("\nSimulating error detection events...")

    # Example: Two Z-syndrome detections (indicating X errors)
    z_detections = {(2, 1), (4, 3)}

    print(f"Z-syndrome detections (X errors): {z_detections}")

    # Decode
    x_chain = decoder.mwpm.decode_x_errors(z_detections)

    print(f"\nDecoded X error chain:")
    print(f"  Correction positions: {x_chain.data_qubit_positions}")
    print(f"  Weight: {x_chain.weight}")

    # Example: Single X-syndrome detection (indicating Z error)
    x_detections = {(1, 2)}

    print(f"\nX-syndrome detections (Z errors): {x_detections}")

    z_chain = decoder.mwpm.decode_z_errors(x_detections)

    print(f"\nDecoded Z error chain:")
    print(f"  Correction positions: {z_chain.data_qubit_positions}")
    print(f"  Weight: {z_chain.weight}")

    # Generate corrections
    corrections = decoder.apply_corrections(x_chain, z_chain)

    print(f"\nTotal correction operations: {sum(len(ops) for ops in corrections.values())}")
    print(f"Data qubits requiring correction: {len(corrections)}")

    print("\n" + "=" * 80)
    print("Decoder demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_decoder()
