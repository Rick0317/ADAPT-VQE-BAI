#!/usr/bin/env python3
"""
ADAPT-VQE using Anastasiou O(N^5) gradient method.

This script runs ADAPT-VQE with the efficient gradient computation from
Anastasiou et al. 2023 (arXiv:2306.03227).

Usage:
    python run_anastasiou_adapt_vqe.py <mol_file> <mol> <n_qubits> <n_electrons> <pool_type> [grad_tol] [max_iter]

Example:
    python run_anastasiou_adapt_vqe.py h4_sto-3g.pkl h4 8 4 qubit_pool 1e-4 50
"""

import numpy as np
import pickle
import sys
import os
import gc
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit.quantum_info import Statevector
from scipy.sparse.linalg import expm_multiply
from openfermion.linalg import qubit_operator_sparse

from adaptvqe.adapt_vqe_preparation import (
    openfermion_qubitop_to_sparsepauliop,
    exact_ground_state_energy,
    create_ansatz_statevector,
    measure_expectation_statevector,
    save_results_to_csv,
    save_intermediate_results_to_csv
)
from utils.ferm_utils import ferm_to_qubit
from utils.reference_state_utils import get_reference_state, get_occ_no
from utils.cnot_counting import count_cnots_from_qubit_operator
from get_generator_pool import get_generator_pool
from sparse_energy_calculation import sparse_multi_parameter_energy_optimization

from anastasiou_gradient import (
    AnastasiouGradientComputer,
    compute_anastasiou_gradient_all_pool,
    bai_find_best_arm_anastasiou
)


def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def print_memory_usage(label=""):
    """Print current memory usage with optional label"""
    memory_mb = get_memory_usage()
    if memory_mb > 0:
        print(f"Memory usage {label}: {memory_mb:.1f} MB")
    return memory_mb


def adapt_vqe_anastasiou(H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op,
                         generator_pool, grad_tol=1e-4, max_iter=50,
                         verbose=True, mol='h4', save_intermediate=True,
                         intermediate_filename='adapt_vqe_anastasiou_intermediate.csv',
                         exact_energy=None):
    """
    ADAPT-VQE algorithm using Anastasiou O(N^5) gradient method.

    Args:
        H_sparse_pauli_op: Hamiltonian as SparsePauliOp
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        H_qubit_op: Hamiltonian as QubitOperator
        generator_pool: List of pool operators
        grad_tol: Gradient convergence threshold
        max_iter: Maximum iterations
        verbose: Print progress
        mol: Molecule name
        save_intermediate: Save results after each iteration
        intermediate_filename: Filename for intermediate results
        exact_energy: Exact ground state energy (computed if not provided)

    Returns:
        tuple: (energies, params, ansatz_ops, final_state, total_measurements, total_cnot_count)
    """
    # Initialize
    ansatz_ops = []
    ansatz_qubit_ops = []
    params = []
    energies = []
    total_measurements = 0
    total_measurements_at_each_step = []
    total_cnot_count = 0

    # Compute exact energy if not provided
    if exact_energy is None:
        H_sparse = H_sparse_pauli_op.to_matrix(sparse=True)
        exact_energy, _ = exact_ground_state_energy(H_sparse)
        print(f"Exact ground state energy: {exact_energy:.8f}")

    # Create initial HF statevector
    initial_state = create_ansatz_statevector(n_qubits, n_electrons, [], [], mol=mol)
    current_statevector = initial_state

    # Measure initial energy
    energy = measure_expectation_statevector(current_statevector, H_sparse_pauli_op)
    print(f"Initial HF energy: {energy:.8f}")

    # Initialize Anastasiou gradient computer (precomputes pool partitioning)
    print("\nInitializing Anastasiou gradient computer...")
    gradient_computer = AnastasiouGradientComputer(generator_pool, n_qubits)

    # Print scaling info
    scaling = gradient_computer.get_measurement_scaling(H_qubit_op)
    print(f"\nMeasurement scaling:")
    print(f"  H terms: {scaling['n_h_terms']}")
    print(f"  Pool size: {scaling['n_pool_ops']}")
    print(f"  Anchor sets: {scaling['n_anchor_sets']}")
    print(f"  Theoretical O(N^5) measurements: {scaling['theoretical_measurements']}")
    print(f"  Naive O(N^6+) measurements: {scaling['naive_measurements']}")
    print(f"  Speedup factor: {scaling['speedup_factor']:.1f}x")

    print("\n" + "="*60)
    print("Starting ADAPT-VQE with Anastasiou gradient method")
    print("="*60)

    for iteration in range(max_iter):
        print(f"\n--- Iteration {iteration} ---")
        print(f"Ansatz depth: {len(ansatz_ops)}, Parameters: {[f'{p:.4f}' for p in params]}")

        # Compute all gradients using Anastasiou method
        print("Computing gradients...")
        gradients, iter_measurements = gradient_computer.compute_all_gradients(
            current_statevector, H_qubit_op
        )

        total_measurements += iter_measurements
        total_measurements_at_each_step.append(total_measurements)

        # Find best operator
        abs_gradients = np.abs(gradients)
        best_idx = int(np.argmax(abs_gradients))
        max_grad = float(abs_gradients[best_idx])

        if verbose:
            print(f"Max gradient: {max_grad:.6e} (operator {best_idx})")
            print(f"Measurements this iteration: {iter_measurements}")

        # Check convergence
        if max_grad < grad_tol:
            print(f"\nConverged: gradient {max_grad:.6e} < threshold {grad_tol}")
            break

        # Add best operator to ansatz
        ansatz_qubit_ops.append(generator_pool[best_idx])
        ansatz_ops.append(qubit_operator_sparse(generator_pool[best_idx], n_qubits))

        # Count CNOTs
        new_op_cnots = count_cnots_from_qubit_operator(generator_pool[best_idx])
        total_cnot_count += new_op_cnots
        print(f"CNOTs for operator {best_idx}: {new_op_cnots}, Total: {total_cnot_count}")

        # Initialize new parameter
        initial_param = 0.01 if gradients[best_idx] > 0 else -0.01
        params.append(initial_param)

        # Optimize all parameters
        print("Optimizing parameters...")
        H_sparse_matrix = H_sparse_pauli_op.to_matrix(sparse=True)

        optimal_params, optimal_energy, updated_state = sparse_multi_parameter_energy_optimization(
            H_sparse_matrix, initial_state.data, ansatz_ops, params,
            max_iter=30, tol=1e-6
        )

        # Update parameters and state
        if hasattr(optimal_params, 'tolist'):
            params = optimal_params.tolist()
        else:
            params = list(optimal_params)

        energy = optimal_energy
        current_statevector = Statevector(updated_state)
        energies.append(energy)

        # Print progress
        energy_error = energy - exact_energy
        print(f"Energy: {energy:.8f} (error: {energy_error:.6e})")

        # Check chemical accuracy
        if abs(energy_error) < 0.0016:  # 1 kcal/mol = 0.0016 Hartree
            print(f"\nReached chemical accuracy!")

        if abs(energy_error) < 0.0001:
            print(f"\nReached target accuracy (0.1 mHartree)")
            break

        # Save intermediate results
        if save_intermediate:
            save_intermediate_results_to_csv(
                iteration=iteration,
                energy=energy,
                params=params,
                ansatz_depth=len(ansatz_ops),
                total_measurements=total_measurements,
                exact_energy=exact_energy,
                molecule_name=mol,
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                pool_size=len(generator_pool),
                use_parallel=False,
                executor_type='anastasiou',
                max_workers=1,
                total_measurements_at_each_step=total_measurements_at_each_step,
                total_measurements_trend_bai={},
                N_est=[0, 0, 0],
                best_idx=best_idx,
                total_cnot_count=total_cnot_count,
                filename=intermediate_filename
            )

        # Memory cleanup
        del H_sparse_matrix, updated_state
        gc.collect()

    # Create final statevector
    final_state = create_ansatz_statevector(n_qubits, n_electrons, ansatz_ops, params, mol=mol)

    print("\n" + "="*60)
    print("ADAPT-VQE COMPLETED")
    print("="*60)
    print(f"Final energy: {energies[-1] if energies else energy:.8f}")
    print(f"Exact energy: {exact_energy:.8f}")
    print(f"Energy error: {(energies[-1] if energies else energy) - exact_energy:.6e}")
    print(f"Ansatz depth: {len(ansatz_ops)}")
    print(f"Total measurements: {total_measurements}")
    print(f"Total CNOTs: {total_cnot_count}")

    return energies, params, ansatz_ops, final_state, total_measurements, total_measurements_at_each_step, total_cnot_count


def main():
    if len(sys.argv) < 6:
        print("Usage: python run_anastasiou_adapt_vqe.py <mol_file> <mol> <n_qubits> <n_electrons> <pool_type> [grad_tol] [max_iter]")
        print("\nExample:")
        print("  python run_anastasiou_adapt_vqe.py h4_sto-3g.pkl h4 8 4 qubit_pool 1e-4 50")
        print("\nAvailable pool types: qubit_pool, uccsd, spin_complemented_uccsd")
        sys.exit(1)

    mol_file = sys.argv[1]
    mol = sys.argv[2]
    n_qubits = int(sys.argv[3])
    n_electrons = int(sys.argv[4])
    pool_type = sys.argv[5]
    grad_tol = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-4
    max_iter = int(sys.argv[7]) if len(sys.argv) > 7 else 50

    print("="*60)
    print("ADAPT-VQE with Anastasiou O(N^5) Gradient Method")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Molecule: {mol}")
    print(f"  Hamiltonian file: {mol_file}")
    print(f"  Qubits: {n_qubits}")
    print(f"  Electrons: {n_electrons}")
    print(f"  Pool type: {pool_type}")
    print(f"  Gradient tolerance: {grad_tol}")
    print(f"  Max iterations: {max_iter}")

    # Load Hamiltonian
    print(f"\nLoading Hamiltonian from ham_lib/{mol_file}...")
    ham_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ham_lib', mol_file)
    with open(ham_path, 'rb') as f:
        fermion_op = pickle.load(f)

    H_qubit_op = ferm_to_qubit(fermion_op)
    H_sparse_pauli_op = openfermion_qubitop_to_sparsepauliop(H_qubit_op, n_qubits)

    # Compute exact energy
    H_sparse = H_sparse_pauli_op.to_matrix(sparse=True)
    exact_energy, exact_gs = exact_ground_state_energy(H_sparse)
    print(f"Exact ground state energy: {exact_energy:.8f}")

    # Create generator pool
    generator_pool = get_generator_pool(pool_type, n_qubits, n_electrons)
    print(f"Generator pool size: {len(generator_pool)}")

    # Setup output filename
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), mol)
    os.makedirs(output_dir, exist_ok=True)
    intermediate_filename = os.path.join(output_dir, f'adapt_vqe_anastasiou_{pool_type}_{time_string}.csv')

    # Run ADAPT-VQE
    print_memory_usage("before ADAPT-VQE")

    energies, params, ansatz_ops, final_state, total_measurements, measurements_at_steps, total_cnot_count = adapt_vqe_anastasiou(
        H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op, generator_pool,
        grad_tol=grad_tol, max_iter=max_iter, mol=mol, exact_energy=exact_energy,
        save_intermediate=True, intermediate_filename=intermediate_filename
    )

    print_memory_usage("after ADAPT-VQE")

    # Calculate fidelity
    if exact_gs is not None:
        overlap = np.abs(np.vdot(final_state.data, exact_gs)) ** 2
        print(f"Fidelity with exact ground state: {overlap:.8f}")
    else:
        overlap = 0.0

    # Save final results
    final_filename = os.path.join(output_dir, f'adapt_vqe_anastasiou_{pool_type}_final_{time_string}.csv')
    save_results_to_csv(
        final_energy=energies[-1] if energies else 0.0,
        energy_at_each_step=energies,
        total_measurements=total_measurements,
        exact_energy=exact_energy,
        fidelity=overlap,
        molecule_name=mol,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        pool_size=len(generator_pool),
        use_parallel=False,
        executor_type='anastasiou',
        max_workers=1,
        ansatz_depth=len(ansatz_ops),
        total_measurements_at_each_step=measurements_at_steps,
        total_measurements_trend_bai={},
        total_cnot_count=total_cnot_count,
        filename=final_filename
    )

    print(f"\nResults saved to: {final_filename}")


if __name__ == "__main__":
    main()
