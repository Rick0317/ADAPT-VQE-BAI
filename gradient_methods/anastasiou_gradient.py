"""
Implementation of O(N^5) gradient computation from Anastasiou et al. 2023
arXiv:2306.03227: "How to really measure operator gradients in ADAPT-VQE"

CORRECT APPROACH (Section III.C):
- Group POOL operators (not Hamiltonian!) into 2N anchor-based commuting sets
- YXXX operators: grouped by Y position (anchor) -> N sets
- XYYY operators: grouped by X position (anchor) -> N sets

Key theorem (Section III.A):
If S_i and S_j commute, then [P, S_i] and [P, S_j] also commute for any pivot P.
[[P, S_i], [P, S_j]] = 4[S_j, S_i] = 0 when S_i, S_j commute

Algorithm:
1. Partition pool into 2N anchor-based commuting sets
2. For each Hamiltonian term H_j (pivot):
   - Measure [H_j, A_i] for all pool operators using 2N sets
   - Commutators in same set can be measured simultaneously

Scaling: O(N^4) H terms × 2N pool sets = O(N^5)
"""

import numpy as np
from openfermion import QubitOperator
from qiskit.quantum_info import SparsePauliOp, Statevector
from typing import List, Tuple, Dict, Set
import gc

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptvqe.adapt_vqe_preparation import openfermion_qubitop_to_sparsepauliop


def get_pauli_structure(pauli_string: tuple) -> Dict[str, List[int]]:
    """
    Analyze Pauli string structure to get X, Y, Z positions.

    Args:
        pauli_string: tuple of (qubit_idx, pauli_type) pairs

    Returns:
        dict with 'X', 'Y', 'Z' keys mapping to lists of qubit indices
    """
    structure = {'X': [], 'Y': [], 'Z': []}
    for qubit, pauli in pauli_string:
        if pauli in structure:
            structure[pauli].append(qubit)
    return structure


def classify_qubit_pool_operator(op: QubitOperator) -> List[Tuple[str, int, tuple]]:
    """
    Classify a qubit pool operator by its Pauli structure.

    Qubit pool operators (after Z-string removal) are:
    - Single excitations: iYX (1 Y, 1 X)
    - Double excitations: iYXXX (1 Y, 3 X) or iXYYY (1 X, 3 Y)

    Args:
        op: QubitOperator from qubit pool

    Returns:
        List of (type, anchor, pauli_string) tuples for each term
        type: 'yxxx', 'xyyy', 'yx' (single)
        anchor: the fixed qubit index for grouping
    """
    classifications = []

    for pauli_string, coeff in op.terms.items():
        if not pauli_string:
            continue

        structure = get_pauli_structure(pauli_string)
        n_x = len(structure['X'])
        n_y = len(structure['Y'])
        n_z = len(structure['Z'])

        # Skip if has Z (shouldn't happen in qubit pool after Z-removal)
        if n_z > 0:
            continue

        # YXXX type: 1 Y, 3 X -> anchor on Y position
        if n_y == 1 and n_x == 3:
            anchor = structure['Y'][0]
            classifications.append(('yxxx', anchor, pauli_string))

        # XYYY type: 1 X, 3 Y -> anchor on X position
        elif n_x == 1 and n_y == 3:
            anchor = structure['X'][0]
            classifications.append(('xyyy', anchor, pauli_string))

        # Single excitation YX: 1 Y, 1 X -> can use either as anchor
        elif n_y == 1 and n_x == 1:
            # For singles, we add to both YXXX (anchor on Y) and XYYY (anchor on X)
            classifications.append(('yx_y', structure['Y'][0], pauli_string))
            classifications.append(('yx_x', structure['X'][0], pauli_string))

        # Other structures (shouldn't occur in standard qubit pool)
        else:
            # Generic fallback - use first non-identity position
            all_positions = structure['X'] + structure['Y']
            if all_positions:
                classifications.append(('other', all_positions[0], pauli_string))

    return classifications


def partition_qubit_pool_into_anchor_sets(generator_pool: List[QubitOperator],
                                          n_qubits: int) -> Dict[str, Dict[int, Set[int]]]:
    """
    Partition qubit pool operators into 2N anchor-based commuting sets.

    Per Section III.C of Anastasiou et al.:
    - YXXX operators: grouped by Y position (anchor) -> N sets
    - XYYY operators: grouped by X position (anchor) -> N sets
    - Single excitations (YX): can be added to both types

    Args:
        generator_pool: List of QubitOperator objects
        n_qubits: Number of qubits

    Returns:
        dict: {'yxxx': {anchor: set(pool_indices)}, 'xyyy': {anchor: set(pool_indices)}}
    """
    # Initialize empty sets for each anchor position
    yxxx_sets = {i: set() for i in range(n_qubits)}
    xyyy_sets = {i: set() for i in range(n_qubits)}

    for pool_idx, op in enumerate(generator_pool):
        classifications = classify_qubit_pool_operator(op)

        for op_type, anchor, pauli_string in classifications:
            if op_type == 'yxxx':
                yxxx_sets[anchor].add(pool_idx)
            elif op_type == 'xyyy':
                xyyy_sets[anchor].add(pool_idx)
            elif op_type == 'yx_y':
                yxxx_sets[anchor].add(pool_idx)
            elif op_type == 'yx_x':
                xyyy_sets[anchor].add(pool_idx)
            elif op_type == 'other':
                # Put in both for safety
                yxxx_sets[anchor].add(pool_idx)
                xyyy_sets[anchor].add(pool_idx)

    return {'yxxx': yxxx_sets, 'xyyy': xyyy_sets}


def compute_pauli_commutator(h_term: tuple, h_coeff: complex,
                             pool_op: QubitOperator) -> QubitOperator:
    """
    Compute commutator [H_j, A_i] where H_j is a single Pauli term.

    Args:
        h_term: tuple of (qubit_idx, pauli_type) for Hamiltonian term
        h_coeff: coefficient of Hamiltonian term
        pool_op: Pool operator A_i

    Returns:
        QubitOperator representing [H_j, A_i]
    """
    H_j = QubitOperator(h_term, h_coeff)
    return H_j * pool_op - pool_op * H_j


def measure_operator_expectation(statevector: Statevector,
                                 op: QubitOperator,
                                 n_qubits: int) -> complex:
    """
    Measure expectation value of a QubitOperator.

    Args:
        statevector: Current quantum state
        op: QubitOperator to measure
        n_qubits: Number of qubits

    Returns:
        complex: Expectation value
    """
    if not op.terms:
        return 0.0

    sparse_op = openfermion_qubitop_to_sparsepauliop(op, n_qubits)
    return statevector.expectation_value(sparse_op)


class AnastasiouGradientComputer:
    """
    Implements efficient gradient computation for ADAPT-VQE using
    the Anastasiou et al. 2023 method (arXiv:2306.03227).

    Key optimization: Partition pool operators into 2N anchor-based
    commuting sets, allowing simultaneous measurement of commutators
    with each Hamiltonian term.
    """

    def __init__(self, generator_pool: List[QubitOperator], n_qubits: int):
        """
        Initialize with generator pool (precompute partitioning).

        Args:
            generator_pool: List of QubitOperator pool operators
            n_qubits: Number of qubits
        """
        self.pool = generator_pool
        self.n_qubits = n_qubits
        self.pool_size = len(generator_pool)

        # Precompute pool partitioning into 2N anchor sets
        print(f"  Partitioning {self.pool_size} pool operators into 2N={2*n_qubits} anchor sets...")
        self.pool_sets = partition_qubit_pool_into_anchor_sets(generator_pool, n_qubits)

        # Count non-empty sets
        n_yxxx = sum(1 for s in self.pool_sets['yxxx'].values() if s)
        n_xyyy = sum(1 for s in self.pool_sets['xyyy'].values() if s)
        print(f"  Non-empty sets: {n_yxxx} YXXX, {n_xyyy} XYYY")

    def compute_all_gradients(self, statevector: Statevector,
                              H_qubit_op: QubitOperator,
                              epsilon: float = 0.01) -> Tuple[np.ndarray, int]:
        """
        Compute all pool gradients using Anastasiou method.

        For each Hamiltonian term H_j (pivot):
        - Measure [H_j, A_i] for all pool operators
        - Group by 2N anchor sets for simultaneous measurement

        Args:
            statevector: Current quantum state |psi>
            H_qubit_op: Hamiltonian as QubitOperator
            epsilon: Target accuracy (for shot allocation)

        Returns:
            gradients: Array of gradient values for each pool operator
            total_measurements: Total number of measurement sets used
        """
        gradients = np.zeros(self.pool_size)
        total_measurement_sets = 0

        # Count non-trivial Hamiltonian terms
        h_terms = [(term, coeff) for term, coeff in H_qubit_op.terms.items()
                   if term and abs(coeff) > 1e-12]
        n_h_terms = len(h_terms)

        print(f"  Computing gradients: {n_h_terms} H terms × 2N pool sets...")

        # For each Hamiltonian term as pivot
        for h_idx, (h_term, h_coeff) in enumerate(h_terms):
            if (h_idx + 1) % 100 == 0:
                print(f"    Processing H term {h_idx + 1}/{n_h_terms}...")

            # Process each of the 2N anchor sets
            for set_type in ['yxxx', 'xyyy']:
                for anchor, pool_indices in self.pool_sets[set_type].items():
                    if not pool_indices:
                        continue

                    # All commutators [H_j, A_i] for A_i in this set can be
                    # measured simultaneously (they commute by Theorem III.A)
                    for pool_idx in pool_indices:
                        # Compute commutator [H_j, A_i]
                        commutator = compute_pauli_commutator(
                            h_term, h_coeff, self.pool[pool_idx]
                        )

                        # Measure expectation <psi|[H_j, A_i]|psi>
                        if commutator.terms:
                            exp_val = measure_operator_expectation(
                                statevector, commutator, self.n_qubits
                            )
                            gradients[pool_idx] += np.real(exp_val)

                    # Count as one simultaneous measurement for this set
                    total_measurement_sets += 1

        # Clean up
        gc.collect()

        return gradients, total_measurement_sets

    def get_measurement_scaling(self, H_qubit_op: QubitOperator) -> Dict[str, int]:
        """
        Compute theoretical measurement scaling.

        Returns:
            dict with scaling information
        """
        n_h_terms = sum(1 for term, coeff in H_qubit_op.terms.items()
                        if term and abs(coeff) > 1e-12)
        n_nonempty_sets = sum(1 for s in self.pool_sets['yxxx'].values() if s)
        n_nonempty_sets += sum(1 for s in self.pool_sets['xyyy'].values() if s)

        return {
            'n_qubits': self.n_qubits,
            'n_h_terms': n_h_terms,
            'n_pool_ops': self.pool_size,
            'n_anchor_sets': n_nonempty_sets,
            'theoretical_measurements': n_h_terms * n_nonempty_sets,
            'naive_measurements': n_h_terms * self.pool_size,
            'speedup_factor': self.pool_size / max(n_nonempty_sets, 1)
        }


def compute_anastasiou_gradient_all_pool(statevector: Statevector,
                                         H_qubit_op: QubitOperator,
                                         generator_pool: List[QubitOperator],
                                         n_qubits: int,
                                         epsilon: float = 0.01) -> Tuple[np.ndarray, int]:
    """
    Compute gradients for all pool operators using Anastasiou method.

    This is the main entry point for the corrected O(N^5) method.

    Args:
        statevector: Current quantum state
        H_qubit_op: Hamiltonian
        generator_pool: List of pool operators
        n_qubits: Number of qubits
        epsilon: Target accuracy

    Returns:
        gradients: Array of gradient values
        total_measurements: Total measurement sets used
    """
    computer = AnastasiouGradientComputer(generator_pool, n_qubits)

    # Print scaling info
    scaling = computer.get_measurement_scaling(H_qubit_op)
    print(f"  Measurement scaling:")
    print(f"    H terms: {scaling['n_h_terms']}")
    print(f"    Pool size: {scaling['n_pool_ops']}")
    print(f"    Anchor sets: {scaling['n_anchor_sets']}")
    print(f"    Theoretical measurements: {scaling['theoretical_measurements']}")
    print(f"    Naive measurements: {scaling['naive_measurements']}")
    print(f"    Speedup factor: {scaling['speedup_factor']:.1f}x")

    gradients, total_measurements = computer.compute_all_gradients(
        statevector, H_qubit_op, epsilon
    )

    del computer
    gc.collect()

    return gradients, total_measurements


def bai_find_best_arm_anastasiou(statevector: Statevector,
                                 H_qubit_op: QubitOperator,
                                 generator_pool: List[QubitOperator],
                                 n_qubits: int,
                                 target_accuracy: float = 0.001) -> Tuple[float, int, int]:
    """
    Find the best arm (operator with largest gradient) using Anastasiou method.

    Args:
        statevector: Current quantum state
        H_qubit_op: Hamiltonian
        generator_pool: List of pool operators
        n_qubits: Number of qubits
        target_accuracy: Target accuracy

    Returns:
        max_gradient: Maximum absolute gradient value
        best_idx: Index of best operator
        total_measurements: Total measurements used
    """
    gradients, total_measurements = compute_anastasiou_gradient_all_pool(
        statevector, H_qubit_op, generator_pool, n_qubits, target_accuracy
    )

    # Find operator with largest absolute gradient
    abs_gradients = np.abs(gradients)
    best_idx = int(np.argmax(abs_gradients))
    max_gradient = float(abs_gradients[best_idx])

    print(f"  Anastasiou BAI: Best operator {best_idx} with gradient {max_gradient:.6e}")

    return max_gradient, best_idx, total_measurements
