from adaptvqe.adapt_vqe_preparation import (create_ansatz_circuit, measure_expectation,
                              get_statevector, openfermion_qubitop_to_sparsepauliop,
                              exact_ground_state_energy, save_results_to_csv, save_intermediate_results_to_csv,
                              create_ansatz_statevector, measure_expectation_statevector,
                              get_hf_statevector, apply_operator_to_statevector)
from scipy.optimize import minimize
import numpy as np
from multiprocessing import Pool, cpu_count, get_context
import gc
from qiskit.quantum_info import SparsePauliOp
from utils.qubit_utils import get_commutator_qubit, qubit_operator_to_qiskit_operator
from adapt_vqe_qiskit_qubitwise_bai import compute_commutator_gradient, generate_commutator_cache_filename, get_commutator_maps_cached
from utils.reference_state_utils import get_reference_state, get_occ_no
from utils.ferm_utils import ferm_to_qubit
import pickle
import os
import sys
import numpy as np
from get_generator_pool import get_generator_pool
from utils.cnot_counting import count_cnots_from_qubit_operator
from datetime import datetime
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import Statevector
from sparse_energy_calculation import sparse_multi_parameter_energy_optimization
from openfermion.linalg import qubit_operator_sparse
# Memory tracking imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with 'pip install psutil' for memory tracking.")

def get_memory_usage():
    """Get current memory usage in MB"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    else:
        return 0.0

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    if PSUTIL_AVAILABLE:
        # Force garbage collection multiple times
        for _ in range(5):
            gc.collect()

        # Try to release memory back to OS
        try:
            import ctypes
            libc = ctypes.CDLL("libc.dylib")
            libc.malloc_trim(0)
        except:
            pass

        # Try to clear NumPy memory pools
        try:
            import numpy as np
            np._no_nep50_warning = True
            # Force NumPy to release memory pools
            np.zeros(1)  # Trigger memory pool cleanup
        except:
            pass

        # Print memory after cleanup
        memory_mb = get_memory_usage()
        print(f"Memory after forced cleanup: {memory_mb:.1f} MB")
        return memory_mb
    return 0.0

def print_memory_usage(label=""):
    """Print current memory usage with optional label"""
    memory_mb = get_memory_usage()
    print(f"Memory usage {label}: {memory_mb:.1f} MB")
    return memory_mb

def track_memory_usage(func):
    """Decorator to track memory usage before and after function execution"""
    def wrapper(*args, **kwargs):
        if PSUTIL_AVAILABLE:
            memory_before = get_memory_usage()
            print(f"Memory before {func.__name__}: {memory_before:.1f} MB")

            result = func(*args, **kwargs)

            memory_after = get_memory_usage()
            memory_diff = memory_after - memory_before
            print(f"Memory after {func.__name__}: {memory_after:.1f} MB (diff: {memory_diff:+.1f} MB)")

            return result
        else:
            return func(*args, **kwargs)
    return wrapper



def compute_exact_commutator_gradient_with_statevector(current_statevector, H_qubit_op, generator_op, n_qubits, radius):
    """
    Ultra-fast computation of energy gradient using sparse matrix operations and optimized Pauli calculations
    Uses statevector directly instead of circuit.
    """

    # Use the statevector directly
    state_vector = current_statevector.data  # Convert to numpy array

    # Validate state vector
    if not np.all(np.isfinite(state_vector)):
        print(f"    Warning: State vector contains non-finite values (NaN/inf)")
        return 0.0, 1.0, [1.0, 1.0, 1.0], 0

    # Normalize state vector to prevent numerical issues
    state_norm = np.linalg.norm(state_vector)
    if state_norm < 1e-10:
        print(f"    Warning: State vector has very small norm: {state_norm}")
        return 0.0, 1.0, [1.0, 1.0, 1.0], 0

    state_vector = state_vector / state_norm

    epsilons = [radius, 0.01, 0.1]

    # Convert to qubit operator
    try:
        commutator_qubit = get_commutator_qubit(H_qubit_op, generator_op)
    except Exception as e:
        print(f"    Error computing commutator: {e}")
        return 0.0, 1.0, [1.0, 1.0, 1.0], 0

    # Decompose into QWC groups
    try:
        from utils.decomposition import qwc_decomposition
        pauli_groups = qwc_decomposition(commutator_qubit)
    except Exception as e:
        print(f"    Error in QWC decomposition: {e}")
        return 0.0, 1.0, [1.0, 1.0, 1.0], 0

    fragment_expectations = []
    fragment_variances = []
    gradient_variance = 0
    total_shots = 0

    for i, group in enumerate(pauli_groups):
        try:
            group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)

            # Vectorized computation for all Pauli strings in this group
            pauli_strings, coeffs = zip(*group_op.to_list())
            coeffs = np.array(coeffs)

            # Validate coefficients
            if not np.all(np.isfinite(coeffs)):
                print(f"    Warning: Non-finite coefficients found in group {i}")
                continue

            # Compute expectations for all Pauli strings in this group at once
            expectations = np.zeros(len(pauli_strings))

            for j, (pauli_string, coeff) in enumerate(zip(pauli_strings, coeffs)):
                try:
                    # Use optimized Pauli expectation calculation
                    expectation = compute_pauli_expectation_fast(state_vector, pauli_string, n_qubits)

                    # Validate expectation value
                    if not np.isfinite(expectation):
                        print(f"    Warning: Non-finite expectation for Pauli string {j} in group {i}: {expectation}")
                        expectation = 0.0

                    expectations[j] = expectation
                except Exception as e:
                    print(f"    Error computing expectation for Pauli string {j} in group {i}: {e}")
                    expectations[j] = 0.0

            # Vectorized fragment computation
            fragment_exact_expectation = np.sum(coeffs * expectations)
            fragment_exact_variance = np.real(np.sum((coeffs**2) * (1.0 - expectations**2)))

            # Validate variance
            if not np.isfinite(fragment_exact_variance) or fragment_exact_variance < 0:
                print(f"    Warning: Invalid variance in group {i}: {fragment_exact_variance}")
                fragment_exact_variance = 1e-12  # Small positive value

            # Ensure variance is not too small to prevent numerical issues
            fragment_exact_variance = max(fragment_exact_variance, 1e-12)

            shots_per_fragment = max(1, int(np.ceil(fragment_exact_variance / radius ** 2)))

            # Sample from normal distribution with safety checks
            try:
                if fragment_exact_variance > 1e-10:
                    fragment_samples = np.real(np.random.normal(
                        fragment_exact_expectation,
                        np.sqrt(fragment_exact_variance),
                        shots_per_fragment
                    ))
                else:
                    # If variance is very small, just use the exact expectation
                    fragment_samples = np.array([fragment_exact_expectation])
            except Exception as e:
                print(f"    Error in normal sampling for group {i}: {e}")
                fragment_samples = np.array([fragment_exact_expectation])

            gradient_variance += fragment_exact_variance

            # Calculate fragment statistics
            fragment_mean = np.mean(fragment_samples)
            if len(fragment_samples) > 1:
                fragment_variance = np.var(fragment_samples, ddof=1)
            else:
                fragment_variance = fragment_exact_variance

            # Validate fragment statistics
            if not np.isfinite(fragment_mean):
                print(f"    Warning: Non-finite fragment mean in group {i}: {fragment_mean}")
                fragment_mean = 0.0
            if not np.isfinite(fragment_variance) or fragment_variance < 0:
                print(f"    Warning: Invalid fragment variance in group {i}: {fragment_variance}")
                fragment_variance = 1e-12

            fragment_expectations.append(fragment_mean)
            fragment_variances.append(fragment_variance)
            total_shots += shots_per_fragment

            # Clean up
            del fragment_samples

        except Exception as e:
            print(f"    Error processing group {i}: {e}")
            continue

    # Validate final results
    if not fragment_expectations:
        print(f"    Warning: No valid fragments computed")
        return 0.0, 1.0, [1.0, 1.0, 1.0], 0

    # Vectorized N_est calculation with safety checks
    try:
        N_est = [max(1.0, gradient_variance / epsilon**2) for epsilon in epsilons]
    except Exception as e:
        print(f"    Error computing N_est: {e}")
        N_est = [1.0, 1.0, 1.0]

    # Estimate total gradient and variance
    estimated_gradient = sum(fragment_expectations)
    estimated_variance = sum(fragment_variances)

    # Final validation of results
    if not np.isfinite(estimated_gradient):
        print(f"    Warning: Non-finite estimated gradient: {estimated_gradient}")
        estimated_gradient = 0.0
    if not np.isfinite(estimated_variance) or estimated_variance < 0:
        print(f"    Warning: Invalid estimated variance: {estimated_variance}")
        estimated_variance = 1e-12

    # Debug: print some information about the computation
    if abs(estimated_gradient) > 1e-10:
        print(f"    Non-zero gradient found: {estimated_gradient:.6e}, variance: {estimated_variance:.6e}")
        print(f"    Number of fragments: {len(fragment_expectations)}")

    # Clean up large objects
    del current_statevector, commutator_qubit, pauli_groups, fragment_expectations, fragment_variances
    del state_vector, expectations, coeffs, pauli_strings
    gc.collect()

    return estimated_gradient, estimated_variance, N_est, total_shots

def compute_pauli_expectation_fast(state_vector, pauli_string, n_qubits):
    """
    Fast Pauli expectation calculation using Qiskit's optimized approach
    """
    try:
        # Validate inputs
        if not np.all(np.isfinite(state_vector)):
            print(f"        Warning: Non-finite state vector in Pauli expectation")
            return 0.0

        if not isinstance(pauli_string, str) or len(pauli_string) != n_qubits:
            print(f"        Warning: Invalid Pauli string: {pauli_string}, n_qubits: {n_qubits}")
            return 0.0

        # Use Qiskit's SparsePauliOp for reliable and fast computation
        pauli_op = SparsePauliOp.from_list([(pauli_string, 1.0)])

        # Convert state vector to Qiskit Statevector for compatibility
        from qiskit.quantum_info import Statevector
        qiskit_state = Statevector(state_vector)

        # Compute expectation value using Qiskit's optimized method
        expectation = qiskit_state.expectation_value(pauli_op)

        # Validate result
        if not np.isfinite(expectation):
            print(f"        Warning: Non-finite expectation value: {expectation}")
            return 0.0

        # Clean up
        del pauli_op, qiskit_state

        return np.real(expectation)

    except Exception as e:
        print(f"        Error in compute_pauli_expectation_fast: {e}")
        return 0.0



def compute_exact_commutator_gradient(current_circuit, H_qubit_op, generator_op, n_qubits, radius):
    """
    Wrapper function that uses the fast implementation
    """
    # Get statevector from circuit
    current_statevector = get_statevector(current_circuit)
    return compute_exact_commutator_gradient_with_statevector(current_statevector, H_qubit_op, generator_op, n_qubits, radius)


def _compute_single_exact_gradient_bai_with_statevector(args):
    """Worker function for parallel exact gradient computation in BAI using statevector directly"""
    try:
        arm_index, current_statevector, H_qubit_op, generator_op, n_qubits, radius = args

        # Compute exact gradient for this arm using statevector directly
        estimate_gradient, estimate_variance, N_est, total_shots = compute_exact_commutator_gradient_with_statevector(
            current_statevector, H_qubit_op, generator_op, n_qubits, radius=radius
        )

        # Clean up and return
        del current_statevector, H_qubit_op, generator_op
        gc.collect()
        return arm_index, estimate_gradient, estimate_variance, N_est, total_shots

    except Exception as e:
        print(f"Error computing exact gradient for arm {arm_index}: {e}")
        gc.collect()
        return arm_index, 0.0, 1.0


@track_memory_usage
def bai_find_the_best_arm_exact_with_statevector(current_statevector, H_qubit_op, generator_pool, n_qubits, x, y, target_accuracy, max_rounds=10, shots_per_round=4096):
    """
    BAI algorithm using exact gradients as means and sampling from normal distributions.
    Computes exact gradients once, then samples from normal distributions for BAI rounds.
    Uses statevector directly instead of circuit.
    """

    K = len(generator_pool)
    active_arms = list(range(K))

    # Store exact gradients and variances
    estimated_gradients = np.zeros(K)
    estimated_variances = np.ones(K)  # Initialize with 1.0

    # Track empirical estimates and pulls for BAI algorithm
    estimates = np.zeros(K)
    exact_N_est = np.zeros(3)
    pulls = np.zeros(K)

    # Get all QWC groups that need to be computed
    # active_qwc_groups = set(fragment_group_indices_map.values())
    rounds = 0
    total_measurements_across_fragments = 0
    measurements_trend_bai = []

    # print(f"Number of active QWC groups: {len(active_qwc_groups)}")

    # First, compute exact gradients for all arms in parallel
    # These will be used as means for normal distribution sampling
    print("Computing exact gradients for all arms...")

    # Check statevector quality before starting gradient computation
    check_statevector_quality(current_statevector, "Before gradient computation")
    accuracy = target_accuracy

    args_list = [
        (i, current_statevector, H_qubit_op, generator_pool[i],
         n_qubits, accuracy)
        for i in active_arms
    ]

    num_processes = min(cpu_count(), len(active_arms))
    shots_across_gradient = 0

    try:
        with Pool(num_processes) as p:
            exact_results = p.map(
                _compute_single_exact_gradient_bai_with_statevector,
                args_list)

        # Store exact gradients and variances
        for arm_index, estimate_gradient, estimate_variance, N_est, total_shots in exact_results:
            # Validate gradient and variance values
            if not np.isfinite(estimate_gradient):
                print(f"    Warning: Non-finite gradient for arm {arm_index}: {estimate_gradient}, setting to 0.0")
                estimate_gradient = 0.0
            if not np.isfinite(estimate_variance) or estimate_variance < 0:
                print(f"    Warning: Invalid variance for arm {arm_index}: {estimate_variance}, setting to 1.0")
                estimate_variance = 1.0

            estimated_gradients[arm_index] = estimate_gradient
            estimated_variances[arm_index] = estimate_variance
            exact_N_est += np.array(N_est, np.float64)
            shots_across_gradient += total_shots

        # Clean up parallel processing results
        del exact_results, args_list
        print_memory_usage("after parallel gradient computation")
        force_memory_cleanup()


    except Exception as e:
        print(
            f"Parallel exact gradient computation failed ({e}), falling back to sequential")
        # Fallback to sequential processing
        for i in active_arms:
            exact_grad, variance, N_est, total_shots = compute_exact_commutator_gradient_with_statevector(
                current_statevector, H_qubit_op, generator_pool[i],
                n_qubits,
                accuracy
            )

            # Validate gradient and variance values
            if not np.isfinite(exact_grad):
                print(f"    Warning: Non-finite gradient for arm {i} (sequential): {exact_grad}, setting to 0.0")
                exact_grad = 0.0
            if not np.isfinite(variance) or variance < 0:
                print(f"    Warning: Invalid variance for arm {i} (sequential): {variance}, setting to 1.0")
                variance = 1.0

            estimated_gradients[i] = exact_grad
            estimated_variances[i] = variance
            exact_N_est += np.array(N_est, np.float64)

            # Clean up after each gradient computation
            gc.collect()
            force_memory_cleanup()


    # Sample from normal distributions using estimated gradients as means
    for i in active_arms:

        # Update running estimates
        if rounds == 1:
            estimates[i] = estimated_gradients[i]
        else:
            estimates[i] = (estimates[i] * pulls[i] + estimated_gradients[i] * shots_per_round) / (pulls[i] + shots_per_round)

        pulls[i] += shots_per_round

    # Final validation of estimates to ensure no NaN values
    for i in active_arms:
        if not np.isfinite(estimates[i]):
            print(f"    Warning: Non-finite estimate for arm {i}: {estimates[i]}, setting to 0.0")
            estimates[i] = 0.0

    total_measurements_across_fragments += shots_across_gradient
    measurements_trend_bai.append(shots_across_gradient)
    # Select best arm based on sampled estimates
    means = estimates
    best_arm = max(np.array(active_arms), key=lambda i: abs(means[i]))
    best_gradient = abs(means[best_arm])

    print(f"Final BAI result: {len(active_arms)} active arms remaining after {rounds} rounds")
    print(f"Selected best arm {best_arm} with sampled gradient magnitude {best_gradient:.6e}")

    return best_gradient, best_arm, total_measurements_across_fragments, measurements_trend_bai, exact_N_est


@track_memory_usage
def fast_energy_calculation(H_sparse_matrix, current_state, new_operator, new_param):
    """
    Fast energy calculation using sparse matrix operations and incremental updates.

    Args:
        H_sparse_matrix: Sparse Hamiltonian matrix
        current_state: Current state vector (numpy array)
        new_operator: New operator to apply
        new_param: Parameter for the new operator

    Returns:
        energy: Energy expectation value
        updated_state: Updated state vector
    """
    # Convert new operator to matrix - avoid sparse conversion for now
    if hasattr(new_operator, 'to_matrix'):
        # Use dense matrix directly to avoid sparse conversion memory leak
        new_op_matrix = new_operator.to_matrix()
    else:
        new_op_matrix = new_operator

    # Apply the new operator using matrix exponential
    if hasattr(new_op_matrix, 'toarray'):  # If it's a sparse matrix
        # Convert to dense for expm
        dense_op = new_op_matrix.toarray()
        updated_state = expm_multiply(new_param * dense_op, current_state)
        del dense_op
    else:  # If it's already a dense matrix
        updated_state = expm_multiply(new_param * new_op_matrix, current_state)

    # Calculate energy using matrix multiplication
    if hasattr(H_sparse_matrix, 'toarray'):  # If it's a sparse matrix
        H_dense = H_sparse_matrix.toarray()
        # Compute matrix-vector product step by step to avoid large temporary arrays
        temp_vector = H_dense @ updated_state
        energy = np.real(np.vdot(updated_state, temp_vector))
        del H_dense, temp_vector
        gc.collect()
    else:  # If it's already a dense matrix
        # Compute matrix-vector product step by step to avoid large temporary arrays
        temp_vector = H_sparse_matrix @ updated_state
        energy = np.real(np.vdot(updated_state, temp_vector))
        del temp_vector
        gc.collect()

    # Clean up intermediate objects
    del new_op_matrix
    gc.collect()
    return energy, updated_state

@track_memory_usage
def adapt_vqe_qiskit(H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op, generator_pool, x, y, target_accuracy, shots=8192, max_iter=100, grad_tol=1e-4, verbose=True, mol='h4', save_intermediate=True, intermediate_filename='adapt_vqe_intermediate_results.csv', exact_energy=None):
    """
    ADAPT-VQE algorithm with multiprocessing support for gradient computation.

    Returns:
        tuple: (energies, params, ansatz_ops, final_state, total_measurements)
    """

    # Prepare reference state
    ansatz_ops = []
    ansatz_qubit_ops = []  # Store original QubitOperators for CNOT counting
    params = []
    energies = []
    total_measurements = 0
    total_measurements_at_each_step = []
    total_measurements_trend_bai = {}
    total_cnot_count = 0

    # Compute exact energy if not provided
    if exact_energy is None:
        H_sparse = H_sparse_pauli_op.to_matrix(sparse=True)
        exact_energy, _ = exact_ground_state_energy(H_sparse)


    initial_state = create_ansatz_statevector(n_qubits, n_electrons,
                                              [],
                                              [], mol=mol)


    final_statevector = initial_state
    energy = measure_expectation_statevector(final_statevector, H_sparse_pauli_op)
    print(f"HF energy (Qiskit): {energy}")

    for iteration in range(max_iter):
        # Use existing statevector instead of reconstructing circuit
        print(f'Iteration {iteration}, Length of Ansatz: {len(ansatz_ops)}, Parameters: {params}')

        # For the first iteration, we need to create the circuit to get the initial statevector
        if iteration == 0:
            current_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops, params)
            # Get the initial statevector
            current_statevector = get_statevector(current_circuit)
            check_statevector_quality(current_statevector, f"Iteration {iteration} - Initial")
        else:
            # For subsequent iterations, use the existing statevector directly
            current_statevector = final_statevector
            check_statevector_quality(current_statevector, f"Iteration {iteration} - Reused")


        max_grad, best_idx, total_measurements_across_fragments, measurements_trend_bai, N_est = (
            bai_find_the_best_arm_exact_with_statevector(current_statevector, H_qubit_op, generator_pool, n_qubits, x, y, target_accuracy, shots_per_round=int(shots)))

        total_measurements += total_measurements_across_fragments
        total_measurements_at_each_step.append(total_measurements)
        total_measurements_trend_bai[iteration] = measurements_trend_bai

        if verbose:
            print(f"Iteration {iteration}: max gradient = {max_grad:.6e}, best = {best_idx}")

        if max_grad < grad_tol:
            if verbose:
                print("Converged: gradient below threshold.")
            break

        # Add best operator to ansatz
        ansatz_qubit_ops.append(generator_pool[best_idx])  # Store original for CNOT counting
        ansatz_ops.append(qubit_operator_sparse(generator_pool[best_idx], n_qubits))

        # Count CNOTs for the newly added operator
        new_op_cnots = count_cnots_from_qubit_operator(generator_pool[best_idx])
        total_cnot_count += new_op_cnots
        print(f"  CNOTs for new operator: {new_op_cnots}, Total CNOTs: {total_cnot_count}")

        print(f"Type of ansazing operators: {type(ansatz_ops[0])}")

        # Use a better initial parameter value based on the gradient
        # If gradient is positive, start with a small positive value; if negative, start with a small negative value
        initial_param = 0.01 if max_grad > 0 else -0.01
        params.append(initial_param)

        # Replace the expensive optimization section with multi-parameter scipy.minimize optimization
        print(f"  Starting multi-parameter scipy.minimize optimization...")

        # Keep Hamiltonian sparse for memory efficiency
        H_sparse_matrix = H_sparse_pauli_op.to_matrix(sparse=True)


        # Use multi-parameter scipy.minimize optimization
        if len(ansatz_ops) > 0:
            optimal_params, optimal_energy, updated_state = sparse_multi_parameter_energy_optimization(
                H_sparse_matrix, initial_state, ansatz_ops, params,
                max_iter=30, tol=1e-6
            )
            # Update all parameters and energy
            # Convert numpy array to list if necessary
            if hasattr(optimal_params, 'tolist'):
                params = optimal_params.tolist()
            elif isinstance(optimal_params, (list, tuple)):
                params = list(optimal_params)
            else:
                params = [float(optimal_params)]  # Handle single parameter case
            energy = optimal_energy
        else:
            # No operators yet, just use the current state and energy
            optimal_params = []
            optimal_energy = energy
            updated_state = initial_state
            # Ensure params remains a list
            params = list(params) if not isinstance(params, list) else params

        # Update final statevector
        final_statevector = Statevector(updated_state)
        check_statevector_quality(final_statevector, f"Iteration {iteration} - After update")
        energies.append(energy)

        if abs(energy- exact_energy) < 0.0001:
            break

        if verbose:
            if len(ansatz_ops) > 0:
                print(f"  Multi-parameter scipy.minimize optimization completed")
                print(f"  Energy after iteration {iteration}: {energy:.8f}")
                print(f"  Optimal parameters: {[f'{p:.6f}' for p in optimal_params]}")
                # Show energy improvement
                if iteration > 0 and len(energies) > 0:
                    energy_improvement = energies[-1] - energy
                    print(f"  Energy improvement: {energy_improvement:.8f}")
            else:
                print(f"  No operators yet, skipping optimization")
                print(f"  Energy after iteration {iteration}: {energy:.8f}")

        # Clean up memory
        del H_sparse_matrix, updated_state
        if len(ansatz_ops) > 0:
            del optimal_params, optimal_energy
        gc.collect()

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Force memory cleanup
        force_memory_cleanup()

        # Save intermediate results after each iteration
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
                use_parallel=True,
                executor_type='multiprocessing',
                max_workers=None,
                total_measurements_at_each_step=total_measurements_at_each_step,
                total_measurements_trend_bai=total_measurements_trend_bai,
                N_est=N_est,
                best_idx=best_idx,
                total_cnot_count=total_cnot_count,
                filename=intermediate_filename
            )

        # Aggressive cleanup after each iteration
        gc.collect()
        force_memory_cleanup()

    # Return final state using fast statevector approach
    final_state = create_ansatz_statevector(n_qubits, n_electrons, ansatz_ops, params, mol=mol)

    if verbose:
        print(f"Total measurements used: {total_measurements}")
        print(f"Total CNOT count: {total_cnot_count}")

    return energies, params, ansatz_ops, final_state, total_measurements, total_measurements_at_each_step, total_measurements_trend_bai, total_cnot_count


def check_statevector_quality(statevector, label=""):
    """Check the quality of a statevector and report any issues"""
    if statevector is None:
        print(f"    {label}: Statevector is None")
        return False

    try:
        state_data = statevector.data
        if not np.all(np.isfinite(state_data)):
            nan_count = np.sum(~np.isfinite(state_data))
            inf_count = np.sum(np.isinf(state_data))
            print(f"    {label}: Statevector contains {nan_count} NaN and {inf_count} inf values")
            return False

        norm = np.linalg.norm(state_data)
        if norm < 1e-10:
            print(f"    {label}: Statevector has very small norm: {norm}")
            return False
        elif norm > 1.1:
            print(f"    {label}: Statevector has large norm: {norm}")
            return False

        print(f"    {label}: Statevector OK - norm: {norm:.6f}, shape: {state_data.shape}")
        return True

    except Exception as e:
        print(f"    {label}: Error checking statevector: {e}")
        return False


if __name__ == "__main__":
    print_memory_usage("at script start")

    if len(sys.argv) < 9:
        print("Usage: python my_script.py <mol_file> <mol> <n_qubits> <n_electrons> <pool_type> <Shots> <x> <y>")
        sys.exit(1)

    mol_file = sys.argv[1]
    # Pre transform all the operators to qubit operators
    # Load H4 Hamiltonian from file
    with open(f'ham_lib/{mol_file}', 'rb') as f:
        fermion_op = pickle.load(f)

    mol = sys.argv[2]
    n_qubits = int(sys.argv[3])
    n_electrons = int(sys.argv[4])
    pool_type = sys.argv[5]
    shots = sys.argv[6]
    x = float(sys.argv[7])
    y = int(sys.argv[8])
    target_accuracy = float(sys.argv[9])

    now = datetime.now()

    # Format as "YYYY-MM-DD HH:MM"
    time_string = now.strftime("%Y-%m-%d%H-%M-%S")

    H_qubit_op = ferm_to_qubit(fermion_op)
    # H_qubit_op = taper_Hamiltonian(H_qubit_op, mol, n_qubits)
    H_sparse_pauli_op = openfermion_qubitop_to_sparsepauliop(H_qubit_op, n_qubits)

    # Compute exact ground state energy
    H_sparse = H_sparse_pauli_op.to_matrix(sparse=True)
    exact_energy, exact_gs = exact_ground_state_energy(H_sparse)
    print(f"Exact ground state energy (diagonalization): {exact_energy:.8f}")

    # Prepare Hartree-Fock state
    ref_occ = get_occ_no(mol, n_qubits)
    hf_state = get_reference_state(ref_occ, gs_format='wfs')

    generator_pool = get_generator_pool(pool_type, n_qubits, n_electrons)
    print(f"Generator pool size: {len(generator_pool)}")


    # Configuration parameters
    use_parallel = True
    max_workers = None
    executor_type = 'multiprocessing'
    molecule_name = mol


    # Intermediate saving configuration
    save_intermediate = True
    intermediate_filename = f'{mol}/adapt_vqe_{pool_type}_results_{time_string}_exact_estimates.csv'

    energies, params, ansatz, final_state, total_measurements, total_measurements_at_each_step, total_measurements_trend_bai, total_cnot_count = adapt_vqe_qiskit(
        H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op, generator_pool,
        x, y, target_accuracy, mol=mol, exact_energy=exact_energy,
        save_intermediate=save_intermediate, intermediate_filename=intermediate_filename)

    # Calculate results
    final_energy = energies[-1]
    overlap = np.abs(np.vdot(final_state.data, exact_gs)) ** 2
    ansatz_depth = len(ansatz)

    # Print results (as before)
    print("Final energy:", final_energy)
    print("Parameters:", params)
    print(f"Ansatz depth: {ansatz_depth}")
    print(f"Total measurements: {total_measurements}")
    print(f"Total CNOT count: {total_cnot_count}")
    print(f"Fidelity (|<ADAPT-VQE|Exact>|^2): {overlap:.8f}")

    # Save results to CSV
    save_results_to_csv(
        final_energy=final_energy,
        energy_at_each_step=energies,
        total_measurements=total_measurements,
        exact_energy=exact_energy,
        fidelity=overlap,
        molecule_name=molecule_name,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        pool_size=len(generator_pool),
        use_parallel=use_parallel,
        executor_type=executor_type,
        max_workers=max_workers,
        ansatz_depth=ansatz_depth,
        total_measurements_at_each_step=total_measurements_at_each_step,
        total_measurements_trend_bai=total_measurements_trend_bai,
        total_cnot_count=total_cnot_count,
        filename=f'adapt_vqe_qubitwise_bai_{mol}_{pool_type}_results.csv'
    )
