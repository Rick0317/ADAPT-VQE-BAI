#!/usr/bin/env python3
"""
Benchmark comparison between standard and Anastasiou gradient methods.

Compares:
- Gradient accuracy (values match)
- Measurement count
- Runtime

Usage:
    python compare_gradient_methods.py <mol_file> <mol> <n_qubits> <n_electrons> <pool_type>

Example:
    python compare_gradient_methods.py h4_sto-3g.pkl h4 8 4 qubit_pool
"""

import time
import numpy as np
import pickle
import sys
import os

from adaptvqe.adapt_vqe_preparation import (
    create_ansatz_circuit, get_statevector, openfermion_qubitop_to_sparsepauliop,
    exact_ground_state_energy, create_ansatz_statevector
)
from adapt_vqe_exact_estimates import compute_exact_commutator_gradient_with_statevector
from gradient_methods.anastasiou_gradient import (
    AnastasiouGradientComputer,
    compute_anastasiou_gradient_all_pool
)
from utils.ferm_utils import ferm_to_qubit
from utils.reference_state_utils import get_reference_state, get_occ_no
from get_generator_pool import get_generator_pool
from qiskit.quantum_info import Statevector


def compare_methods(statevector, H_qubit_op, generator_pool, n_qubits, epsilon=0.01):
    """
    Compare standard vs Anastasiou gradient computation.

    Args:
        statevector: Current quantum state
        H_qubit_op: Hamiltonian as QubitOperator
        generator_pool: List of generator operators
        n_qubits: Number of qubits
        epsilon: Target accuracy

    Returns:
        dict: Results including gradients, measurements, and timing for both methods
    """
    results = {}

    # Standard method
    print("\n" + "="*60)
    print("Running STANDARD method...")
    print("="*60)
    start = time.time()
    standard_grads = []
    standard_vars = []
    standard_total_meas = 0

    for i, gen in enumerate(generator_pool):
        grad, var, N_est, meas = compute_exact_commutator_gradient_with_statevector(
            statevector, H_qubit_op, gen, n_qubits, epsilon
        )
        standard_grads.append(grad)
        standard_vars.append(var)
        standard_total_meas += meas

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(generator_pool)} operators...")

    standard_time = time.time() - start

    results['standard'] = {
        'gradients': np.array(standard_grads),
        'variances': np.array(standard_vars),
        'measurements': standard_total_meas,
        'time': standard_time
    }

    print(f"\nStandard method completed:")
    print(f"  Time: {standard_time:.2f} seconds")
    print(f"  Total measurements: {standard_total_meas}")
    print(f"  Max gradient magnitude: {np.max(np.abs(standard_grads)):.6e}")

    # Anastasiou method
    print("\n" + "="*60)
    print("Running ANASTASIOU method...")
    print("="*60)
    start = time.time()

    anastasiou_grads, anastasiou_meas = compute_anastasiou_gradient_all_pool(
        statevector, H_qubit_op, generator_pool, n_qubits, epsilon
    )

    anastasiou_time = time.time() - start

    results['anastasiou'] = {
        'gradients': anastasiou_grads,
        'measurements': anastasiou_meas,
        'time': anastasiou_time
    }

    print(f"\nAnastasiou method completed:")
    print(f"  Time: {anastasiou_time:.2f} seconds")
    print(f"  Total measurements: {anastasiou_meas}")
    print(f"  Max gradient magnitude: {np.max(np.abs(anastasiou_grads)):.6e}")

    # Analysis
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    grad_diff = np.abs(results['standard']['gradients'] - results['anastasiou']['gradients'])

    # Avoid division by zero
    meas_ratio = results['standard']['measurements'] / max(results['anastasiou']['measurements'], 1)
    time_ratio = results['standard']['time'] / max(results['anastasiou']['time'], 1e-6)

    print(f"\nGradient Accuracy:")
    print(f"  Max absolute difference: {np.max(grad_diff):.6e}")
    print(f"  Mean absolute difference: {np.mean(grad_diff):.6e}")
    print(f"  Median absolute difference: {np.median(grad_diff):.6e}")

    # Check if both methods agree on the best operator
    standard_best = np.argmax(np.abs(results['standard']['gradients']))
    anastasiou_best = np.argmax(np.abs(results['anastasiou']['gradients']))
    print(f"\nBest operator agreement:")
    print(f"  Standard selects: {standard_best} (grad: {results['standard']['gradients'][standard_best]:.6e})")
    print(f"  Anastasiou selects: {anastasiou_best} (grad: {results['anastasiou']['gradients'][anastasiou_best]:.6e})")
    print(f"  Agreement: {'YES' if standard_best == anastasiou_best else 'NO'}")

    print(f"\nEfficiency Comparison:")
    print(f"  Measurement ratio (standard/anastasiou): {meas_ratio:.2f}x")
    print(f"  Time ratio (standard/anastasiou): {time_ratio:.2f}x")

    if meas_ratio > 1:
        print(f"  -> Anastasiou uses {meas_ratio:.1f}x fewer measurements")
    else:
        print(f"  -> Standard uses {1/meas_ratio:.1f}x fewer measurements")

    if time_ratio > 1:
        print(f"  -> Anastasiou is {time_ratio:.1f}x faster")
    else:
        print(f"  -> Standard is {1/time_ratio:.1f}x faster")

    return results


def run_comparison(mol_file, mol, n_qubits, n_electrons, pool_type, epsilon=0.01):
    """
    Run full comparison between gradient methods.

    Args:
        mol_file: Hamiltonian file name
        mol: Molecule name
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        pool_type: Type of operator pool
        epsilon: Target accuracy
    """
    print("="*60)
    print("GRADIENT METHOD COMPARISON")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Molecule: {mol}")
    print(f"  Qubits: {n_qubits}")
    print(f"  Electrons: {n_electrons}")
    print(f"  Pool type: {pool_type}")
    print(f"  Target accuracy: {epsilon}")

    # Load Hamiltonian
    print(f"\nLoading Hamiltonian from ham_lib/{mol_file}...")
    with open(f'ham_lib/{mol_file}', 'rb') as f:
        fermion_op = pickle.load(f)

    H_qubit_op = ferm_to_qubit(fermion_op)
    H_sparse_pauli_op = openfermion_qubitop_to_sparsepauliop(H_qubit_op, n_qubits)

    # Get exact ground state
    H_sparse = H_sparse_pauli_op.to_matrix(sparse=True)
    exact_energy, exact_gs = exact_ground_state_energy(H_sparse)
    print(f"Exact ground state energy: {exact_energy:.8f}")

    # Create generator pool
    generator_pool = get_generator_pool(pool_type, n_qubits, n_electrons)
    print(f"Generator pool size: {len(generator_pool)}")

    # Create initial statevector (Hartree-Fock state)
    print("\nCreating initial HF statevector...")
    initial_statevector = create_ansatz_statevector(n_qubits, n_electrons, [], [], mol=mol)

    # Run comparison
    results = compare_methods(initial_statevector, H_qubit_op, generator_pool, n_qubits, epsilon)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python compare_gradient_methods.py <mol_file> <mol> <n_qubits> <n_electrons> <pool_type> [epsilon]")
        print("\nExample:")
        print("  python compare_gradient_methods.py h4_sto-3g.pkl h4 8 4 qubit_pool 0.01")
        sys.exit(1)

    mol_file = sys.argv[1]
    mol = sys.argv[2]
    n_qubits = int(sys.argv[3])
    n_electrons = int(sys.argv[4])
    pool_type = sys.argv[5]
    epsilon = float(sys.argv[6]) if len(sys.argv) > 6 else 0.01

    results = run_comparison(mol_file, mol, n_qubits, n_electrons, pool_type, epsilon)
