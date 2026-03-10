#!/usr/bin/env python3
"""
ADAPT-VQE using Anastasiou O(N^5) gradient method WITH measurement cost estimation.

This script runs ADAPT-VQE with the efficient gradient computation from
Anastasiou et al. 2023 (arXiv:2306.03227) and estimates the total number
of measurements (shots) required for a given target accuracy.

Measurement estimation follows: N_shots = Var(O) / epsilon^2

Usage:
    python run_anastasiou_adapt_vqe_with_estimates.py <mol_file> <mol> <n_qubits> <n_electrons> <pool_type> [epsilon] [grad_tol] [max_iter]

Example:
    python run_anastasiou_adapt_vqe_with_estimates.py h4_sto-3g.pkl h4 8 4 qubit_pool 0.001 1e-4 50
"""

import numpy as np
import pickle
import sys
import os
import gc
from datetime import datetime
from typing import Tuple, List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit.quantum_info import Statevector, SparsePauliOp
from scipy.sparse.linalg import expm_multiply
from openfermion import QubitOperator
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
from utils.decomposition import qwc_decomposition
from get_generator_pool import get_generator_pool
from sparse_energy_calculation import sparse_multi_parameter_energy_optimization

from anastasiou_gradient import (
    AnastasiouGradientComputer,
    partition_qubit_pool_into_anchor_sets,
    compute_pauli_commutator,
    measure_operator_expectation
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


def compute_operator_variance(statevector: Statevector, op: SparsePauliOp) -> float:
    """
    Compute the variance of an operator measurement: Var(O) = <O^2> - <O>^2

    Args:
        statevector: Current quantum state
        op: Operator as SparsePauliOp

    Returns:
        Variance of the operator
    """
    expectation = np.real(statevector.expectation_value(op))
    op_squared = op @ op
    expectation_squared = np.real(statevector.expectation_value(op_squared))
    variance = expectation_squared - expectation**2
    return max(0.0, variance)  # Ensure non-negative


def estimate_shots_for_accuracy(variance: float, epsilon: float) -> int:
    """
    Estimate number of shots needed for target accuracy.

    Based on: N = Var(O) / epsilon^2

    Args:
        variance: Variance of the measurement
        epsilon: Target accuracy (standard error)

    Returns:
        Estimated number of shots
    """
    if epsilon <= 0:
        return 1
    if variance < 1e-12:
        return 1
    return max(1, int(np.ceil(variance / (epsilon ** 2))))


def compute_hamiltonian_variance(statevector: Statevector,
                                  H_sparse_pauli_op: SparsePauliOp) -> float:
    """
    Compute variance of Hamiltonian measurement for energy estimation.

    Args:
        statevector: Current quantum state
        H_sparse_pauli_op: Hamiltonian as SparsePauliOp

    Returns:
        Variance of energy measurement
    """
    return compute_operator_variance(statevector, H_sparse_pauli_op)


def compute_commutator_variance(statevector: Statevector,
                                 commutator_op: QubitOperator,
                                 n_qubits: int) -> float:
    """
    Compute variance of commutator measurement for gradient estimation.

    Args:
        statevector: Current quantum state
        commutator_op: Commutator [H_j, A_i] as QubitOperator
        n_qubits: Number of qubits

    Returns:
        Variance of the commutator measurement
    """
    if not commutator_op.terms:
        return 0.0

    sparse_op = openfermion_qubitop_to_sparsepauliop(commutator_op, n_qubits)
    return compute_operator_variance(statevector, sparse_op)


class AnastasiouGradientComputerWithEstimates:
    """
    Anastasiou gradient computer that also estimates measurement costs.

    Extends the base implementation to track:
    - Exact gradient values
    - Variance of each gradient measurement
    - Estimated shots needed for target accuracy
    """

    def __init__(self, generator_pool: List[QubitOperator], n_qubits: int):
        self.pool = generator_pool
        self.n_qubits = n_qubits
        self.pool_size = len(generator_pool)

        # Precompute pool partitioning
        print(f"  Partitioning {self.pool_size} pool operators into 2N={2*n_qubits} anchor sets...")
        self.pool_sets = partition_qubit_pool_into_anchor_sets(generator_pool, n_qubits)

        # Count non-empty sets
        n_yxxx = sum(1 for s in self.pool_sets['yxxx'].values() if s)
        n_xyyy = sum(1 for s in self.pool_sets['xyyy'].values() if s)
        print(f"  Non-empty sets: {n_yxxx} YXXX, {n_xyyy} XYYY")

    def compute_all_gradients_with_estimates(self,
                                              statevector: Statevector,
                                              H_qubit_op: QubitOperator,
                                              epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Compute all pool gradients and estimate measurement costs.

        Args:
            statevector: Current quantum state
            H_qubit_op: Hamiltonian as QubitOperator
            epsilon: Target accuracy for gradient estimation

        Returns:
            gradients: Array of exact gradient values
            variances: Array of gradient variances
            total_measurement_sets: Number of simultaneous measurement sets (O(N^5))
            estimated_total_shots: Estimated shots for target accuracy
        """
        gradients = np.zeros(self.pool_size)
        variances = np.zeros(self.pool_size)
        total_measurement_sets = 0

        # For shot estimation, we accumulate variance contributions
        gradient_variance_contributions = [[] for _ in range(self.pool_size)]

        # Get non-trivial Hamiltonian terms
        h_terms = [(term, coeff) for term, coeff in H_qubit_op.terms.items()
                   if term and abs(coeff) > 1e-12]
        n_h_terms = len(h_terms)

        print(f"  Computing gradients with estimates: {n_h_terms} H terms...")

        # For each Hamiltonian term as pivot
        for h_idx, (h_term, h_coeff) in enumerate(h_terms):
            if (h_idx + 1) % 100 == 0:
                print(f"    Processing H term {h_idx + 1}/{n_h_terms}...")

            # Process each anchor set
            for set_type in ['yxxx', 'xyyy']:
                for anchor, pool_indices in self.pool_sets[set_type].items():
                    if not pool_indices:
                        continue

                    # Compute commutators for all operators in this set
                    for pool_idx in pool_indices:
                        # Compute commutator [H_j, A_i]
                        commutator = compute_pauli_commutator(
                            h_term, h_coeff, self.pool[pool_idx]
                        )

                        if commutator.terms:
                            # Measure expectation
                            exp_val = measure_operator_expectation(
                                statevector, commutator, self.n_qubits
                            )
                            gradients[pool_idx] += np.real(exp_val)

                            # Compute variance for this commutator
                            comm_var = compute_commutator_variance(
                                statevector, commutator, self.n_qubits
                            )
                            gradient_variance_contributions[pool_idx].append(comm_var)

                    total_measurement_sets += 1

        # Aggregate variances for each pool operator
        # Total variance is sum of individual variances (assuming independent measurements)
        for i in range(self.pool_size):
            if gradient_variance_contributions[i]:
                variances[i] = sum(gradient_variance_contributions[i])

        # Estimate total shots needed
        estimated_total_shots = 0
        for i in range(self.pool_size):
            shots_for_op = estimate_shots_for_accuracy(variances[i], epsilon)
            estimated_total_shots += shots_for_op

        # Clean up
        gc.collect()

        return gradients, variances, total_measurement_sets, estimated_total_shots

    def get_measurement_scaling(self, H_qubit_op: QubitOperator) -> Dict[str, int]:
        """Compute theoretical measurement scaling."""
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


def estimate_vqe_energy_shots(statevector: Statevector,
                               H_sparse_pauli_op: SparsePauliOp,
                               epsilon: float) -> Tuple[float, float, int]:
    """
    Estimate shots needed for VQE energy measurement.

    Args:
        statevector: Current quantum state
        H_sparse_pauli_op: Hamiltonian
        epsilon: Target accuracy

    Returns:
        energy: Exact energy value
        variance: Variance of energy measurement
        estimated_shots: Estimated shots for target accuracy
    """
    energy = np.real(statevector.expectation_value(H_sparse_pauli_op))
    variance = compute_hamiltonian_variance(statevector, H_sparse_pauli_op)
    estimated_shots = estimate_shots_for_accuracy(variance, epsilon)

    return energy, variance, estimated_shots


def adapt_vqe_anastasiou_with_estimates(H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op,
                                         generator_pool, epsilon=0.001, grad_tol=1e-4, max_iter=50,
                                         verbose=True, mol='h4', save_intermediate=True,
                                         intermediate_filename='adapt_vqe_anastasiou_estimates.csv',
                                         exact_energy=None):
    """
    ADAPT-VQE with Anastasiou gradient method and measurement cost estimation.

    Args:
        H_sparse_pauli_op: Hamiltonian as SparsePauliOp
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        H_qubit_op: Hamiltonian as QubitOperator
        generator_pool: List of pool operators
        epsilon: Target accuracy for measurement estimation
        grad_tol: Gradient convergence threshold
        max_iter: Maximum iterations
        verbose: Print progress
        mol: Molecule name
        save_intermediate: Save results after each iteration
        intermediate_filename: Filename for intermediate results
        exact_energy: Exact ground state energy

    Returns:
        Results tuple with energies, measurements, and estimates
    """
    # Initialize tracking variables
    ansatz_ops = []
    ansatz_qubit_ops = []
    params = []
    energies = []
    total_measurement_sets = 0  # O(N^5) measurement sets
    total_estimated_shots = 0   # Estimated shots for target accuracy
    total_vqe_shots = 0         # Shots for VQE energy measurements
    total_cnot_count = 0

    # Per-iteration tracking
    measurements_at_each_step = []
    estimated_shots_at_each_step = []
    gradient_estimates_per_iter = []  # Track [epsilon=0.001, 0.01, 0.1] estimates

    # Compute exact energy if not provided
    if exact_energy is None:
        H_sparse = H_sparse_pauli_op.to_matrix(sparse=True)
        exact_energy, _ = exact_ground_state_energy(H_sparse)
        print(f"Exact ground state energy: {exact_energy:.8f}")

    # Create initial HF statevector
    initial_state = create_ansatz_statevector(n_qubits, n_electrons, [], [], mol=mol)
    current_statevector = initial_state

    # Measure initial energy with variance
    energy, energy_var, energy_shots = estimate_vqe_energy_shots(
        current_statevector, H_sparse_pauli_op, epsilon
    )
    print(f"Initial HF energy: {energy:.8f}")
    print(f"  Energy variance: {energy_var:.6e}")
    print(f"  Estimated shots (epsilon={epsilon}): {energy_shots}")

    # Initialize gradient computer
    print("\nInitializing Anastasiou gradient computer with estimation...")
    gradient_computer = AnastasiouGradientComputerWithEstimates(generator_pool, n_qubits)

    # Print scaling info
    scaling = gradient_computer.get_measurement_scaling(H_qubit_op)
    print(f"\nMeasurement scaling:")
    print(f"  H terms: {scaling['n_h_terms']}")
    print(f"  Pool size: {scaling['n_pool_ops']}")
    print(f"  Anchor sets: {scaling['n_anchor_sets']}")
    print(f"  Theoretical O(N^5) measurement sets: {scaling['theoretical_measurements']}")
    print(f"  Naive measurement sets: {scaling['naive_measurements']}")
    print(f"  Speedup factor: {scaling['speedup_factor']:.1f}x")

    print("\n" + "="*60)
    print(f"Starting ADAPT-VQE with measurement estimation (epsilon={epsilon})")
    print("="*60)

    for iteration in range(max_iter):
        print(f"\n--- Iteration {iteration} ---")
        print(f"Ansatz depth: {len(ansatz_ops)}, Parameters: {[f'{p:.4f}' for p in params]}")

        # Compute gradients with variance estimation
        print("Computing gradients with variance estimation...")
        gradients, grad_variances, iter_meas_sets, iter_est_shots = \
            gradient_computer.compute_all_gradients_with_estimates(
                current_statevector, H_qubit_op, epsilon
            )

        # Also compute estimates for different epsilon values
        N_est = []
        for eps in [0.001, 0.01, 0.1]:
            est = sum(estimate_shots_for_accuracy(v, eps) for v in grad_variances)
            N_est.append(est)
        gradient_estimates_per_iter.append(N_est)

        total_measurement_sets += iter_meas_sets
        total_estimated_shots += iter_est_shots
        measurements_at_each_step.append(total_measurement_sets)
        estimated_shots_at_each_step.append(total_estimated_shots)

        # Find best operator
        abs_gradients = np.abs(gradients)
        best_idx = int(np.argmax(abs_gradients))
        max_grad = float(abs_gradients[best_idx])
        best_variance = float(grad_variances[best_idx])

        if verbose:
            print(f"Max gradient: {max_grad:.6e} (operator {best_idx})")
            print(f"  Gradient variance: {best_variance:.6e}")
            print(f"  Measurement sets this iteration: {iter_meas_sets}")
            print(f"  Estimated shots this iteration: {iter_est_shots}")
            print(f"  N_est [0.001, 0.01, 0.1]: {N_est}")

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

        # Optimize parameters
        print("Optimizing parameters...")
        H_sparse_matrix = H_sparse_pauli_op.to_matrix(sparse=True)

        optimal_params, optimal_energy, updated_state = sparse_multi_parameter_energy_optimization(
            H_sparse_matrix, initial_state.data, ansatz_ops, params,
            max_iter=30, tol=1e-6
        )

        # Update state
        if hasattr(optimal_params, 'tolist'):
            params = optimal_params.tolist()
        else:
            params = list(optimal_params)

        current_statevector = Statevector(updated_state)

        # Estimate VQE energy measurement cost
        energy, energy_var, vqe_shots = estimate_vqe_energy_shots(
            current_statevector, H_sparse_pauli_op, epsilon
        )
        total_vqe_shots += vqe_shots
        energies.append(energy)

        # Print progress
        energy_error = energy - exact_energy
        print(f"Energy: {energy:.8f} (error: {energy_error:.6e})")
        print(f"  Energy variance: {energy_var:.6e}")
        print(f"  VQE shots estimate: {vqe_shots}")

        # Check accuracy thresholds
        if abs(energy_error) < 0.0016:
            print("Reached chemical accuracy!")
        if abs(energy_error) < 0.0001:
            print("Reached target accuracy (0.1 mHartree)")
            break

        # Save intermediate results
        if save_intermediate:
            save_intermediate_results_to_csv(
                iteration=iteration,
                energy=energy,
                params=params,
                ansatz_depth=len(ansatz_ops),
                total_measurements=total_estimated_shots,  # Use estimated shots
                exact_energy=exact_energy,
                molecule_name=mol,
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                pool_size=len(generator_pool),
                use_parallel=False,
                executor_type='anastasiou_estimates',
                max_workers=1,
                total_measurements_at_each_step=estimated_shots_at_each_step,
                total_measurements_trend_bai={},
                N_est=N_est,
                best_idx=best_idx,
                total_cnot_count=total_cnot_count,
                filename=intermediate_filename
            )

        # Cleanup
        del H_sparse_matrix, updated_state
        gc.collect()

    # Create final state
    final_state = create_ansatz_statevector(n_qubits, n_electrons, ansatz_ops, params, mol=mol)

    # Summary
    print("\n" + "="*60)
    print("ADAPT-VQE WITH ESTIMATION COMPLETED")
    print("="*60)
    print(f"Final energy: {energies[-1] if energies else energy:.8f}")
    print(f"Exact energy: {exact_energy:.8f}")
    print(f"Energy error: {(energies[-1] if energies else energy) - exact_energy:.6e}")
    print(f"Ansatz depth: {len(ansatz_ops)}")
    print(f"\nMeasurement Summary (epsilon={epsilon}):")
    print(f"  Total O(N^5) measurement sets: {total_measurement_sets}")
    print(f"  Estimated gradient shots: {total_estimated_shots}")
    print(f"  Estimated VQE shots: {total_vqe_shots}")
    print(f"  Total estimated shots: {total_estimated_shots + total_vqe_shots}")
    print(f"Total CNOTs: {total_cnot_count}")

    return {
        'energies': energies,
        'params': params,
        'ansatz_ops': ansatz_ops,
        'final_state': final_state,
        'total_measurement_sets': total_measurement_sets,
        'total_estimated_shots': total_estimated_shots,
        'total_vqe_shots': total_vqe_shots,
        'measurements_at_each_step': measurements_at_each_step,
        'estimated_shots_at_each_step': estimated_shots_at_each_step,
        'gradient_estimates_per_iter': gradient_estimates_per_iter,
        'total_cnot_count': total_cnot_count
    }


def main():
    if len(sys.argv) < 6:
        print("Usage: python run_anastasiou_adapt_vqe_with_estimates.py <mol_file> <mol> <n_qubits> <n_electrons> <pool_type> [epsilon] [grad_tol] [max_iter]")
        print("\nExample:")
        print("  python run_anastasiou_adapt_vqe_with_estimates.py h4_sto-3g.pkl h4 8 4 qubit_pool 0.001 1e-4 50")
        print("\nArguments:")
        print("  epsilon: Target accuracy for measurement estimation (default: 0.001)")
        print("  grad_tol: Gradient convergence threshold (default: 1e-4)")
        print("  max_iter: Maximum ADAPT iterations (default: 50)")
        sys.exit(1)

    mol_file = sys.argv[1]
    mol = sys.argv[2]
    n_qubits = int(sys.argv[3])
    n_electrons = int(sys.argv[4])
    pool_type = sys.argv[5]
    epsilon = float(sys.argv[6]) if len(sys.argv) > 6 else 0.001
    grad_tol = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-4
    max_iter = int(sys.argv[8]) if len(sys.argv) > 8 else 50

    print("="*60)
    print("ADAPT-VQE with Anastasiou Method + Measurement Estimation")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Molecule: {mol}")
    print(f"  Hamiltonian file: {mol_file}")
    print(f"  Qubits: {n_qubits}")
    print(f"  Electrons: {n_electrons}")
    print(f"  Pool type: {pool_type}")
    print(f"  Target accuracy (epsilon): {epsilon}")
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

    # Setup output
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), mol)
    os.makedirs(output_dir, exist_ok=True)
    intermediate_filename = os.path.join(output_dir, f'adapt_vqe_anastasiou_estimates_{pool_type}_{time_string}.csv')

    # Run ADAPT-VQE
    print_memory_usage("before ADAPT-VQE")

    results = adapt_vqe_anastasiou_with_estimates(
        H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op, generator_pool,
        epsilon=epsilon, grad_tol=grad_tol, max_iter=max_iter, mol=mol,
        exact_energy=exact_energy, save_intermediate=True,
        intermediate_filename=intermediate_filename
    )

    print_memory_usage("after ADAPT-VQE")

    # Calculate fidelity
    if exact_gs is not None:
        overlap = np.abs(np.vdot(results['final_state'].data, exact_gs)) ** 2
        print(f"Fidelity with exact ground state: {overlap:.8f}")
    else:
        overlap = 0.0

    # Save final results
    final_filename = os.path.join(output_dir, f'adapt_vqe_anastasiou_estimates_{pool_type}_final_{time_string}.csv')
    save_results_to_csv(
        final_energy=results['energies'][-1] if results['energies'] else 0.0,
        energy_at_each_step=results['energies'],
        total_measurements=results['total_estimated_shots'] + results['total_vqe_shots'],
        exact_energy=exact_energy,
        fidelity=overlap,
        molecule_name=mol,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        pool_size=len(generator_pool),
        use_parallel=False,
        executor_type='anastasiou_estimates',
        max_workers=1,
        ansatz_depth=len(results['ansatz_ops']),
        total_measurements_at_each_step=results['estimated_shots_at_each_step'],
        total_measurements_trend_bai={
            'measurement_sets': results['measurements_at_each_step'],
            'gradient_N_est': results['gradient_estimates_per_iter']
        },
        total_cnot_count=results['total_cnot_count'],
        filename=final_filename
    )

    print(f"\nResults saved to: {final_filename}")

    # Print final summary table
    print("\n" + "="*60)
    print("MEASUREMENT COST SUMMARY")
    print("="*60)
    print(f"{'Metric':<40} {'Value':>15}")
    print("-"*60)
    print(f"{'O(N^5) measurement sets':<40} {results['total_measurement_sets']:>15,}")
    print(f"{'Estimated gradient shots (eps={epsilon})':<40} {results['total_estimated_shots']:>15,}")
    print(f"{'Estimated VQE shots (eps={epsilon})':<40} {results['total_vqe_shots']:>15,}")
    print(f"{'Total estimated shots':<40} {results['total_estimated_shots'] + results['total_vqe_shots']:>15,}")
    print(f"{'Total CNOTs':<40} {results['total_cnot_count']:>15,}")
    print("-"*60)


if __name__ == "__main__":
    main()
