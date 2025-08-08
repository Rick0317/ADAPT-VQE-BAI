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
from datetime import datetime
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import Statevector

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


def _compute_single_gradient_bai(args):
    try:
        arm_index, H_qubit_op, generator_op, fragment_indices, counts_for_each_fragment, n_qubits, shots = args
        reward = compute_commutator_gradient(H_qubit_op, generator_op,
                                             fragment_indices,
                                             counts_for_each_fragment, n_qubits,
                                             shots)

        # Clean up and return
        gc.collect()
        return arm_index, reward
    except Exception as e:
        print(f"Error computing gradient for arm {arm_index}: {e}")
        gc.collect()
        return arm_index, 0.0


def compute_exact_commutator_gradient_with_statevector(current_statevector, H_qubit_op, generator_op, fragment_indices, n_qubits, radius):
    """
    Ultra-fast computation of energy gradient using sparse matrix operations and optimized Pauli calculations
    Uses statevector directly instead of circuit.
    """

    # Use the statevector directly
    state_vector = current_statevector.data  # Convert to numpy array

    epsilons = [0.001, 0.01, 0.1]

    # Convert to qubit operator
    commutator_qubit = get_commutator_qubit(H_qubit_op, generator_op)

    # Decompose into QWC groups
    from utils.decomposition import qwc_decomposition
    pauli_groups = qwc_decomposition(commutator_qubit)

    fragment_expectations = []
    fragment_variances = []
    gradient_variance = 0
    total_shots = 0

    for i, group in enumerate(pauli_groups):
        group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)

        # Vectorized computation for all Pauli strings in this group
        pauli_strings, coeffs = zip(*group_op.to_list())
        coeffs = np.array(coeffs)

        # Compute expectations for all Pauli strings in this group at once
        expectations = np.zeros(len(pauli_strings))

        for j, (pauli_string, coeff) in enumerate(zip(pauli_strings, coeffs)):
            # Use optimized Pauli expectation calculation
            expectation = compute_pauli_expectation_fast(state_vector, pauli_string, n_qubits)
            expectations[j] = expectation

        # Vectorized fragment computation
        fragment_exact_expectation = np.sum(coeffs * expectations)
        fragment_exact_variance = np.real(np.sum((coeffs**2) * (1.0 - expectations**2)))
        shots_per_fragment = int(np.ceil(fragment_exact_variance / radius ** 2))

        # Sample from normal distribution


        gradient_variance += fragment_exact_variance

        # Calculate fragment statistics
        fragment_mean = fragment_exact_expectation
        fragment_variance = fragment_exact_variance

        fragment_expectations.append(fragment_mean)
        fragment_variances.append(fragment_variance)
        total_shots += shots_per_fragment


    # Vectorized N_est calculation
    N_est = [gradient_variance / epsilon**2 for epsilon in epsilons]

    # Estimate total gradient and variance
    estimated_gradient = sum(fragment_expectations)
    estimated_variance = sum(fragment_variances)

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
    # Use Qiskit's SparsePauliOp for reliable and fast computation
    pauli_op = SparsePauliOp.from_list([(pauli_string, 1.0)])

    # Convert state vector to Qiskit Statevector for compatibility
    from qiskit.quantum_info import Statevector
    qiskit_state = Statevector(state_vector)

    # Compute expectation value using Qiskit's optimized method
    expectation = qiskit_state.expectation_value(pauli_op)

    return np.real(expectation)

def compute_exact_commutator_gradient(current_circuit, H_qubit_op, generator_op, fragment_indices, n_qubits, shots_per_fragment=1024):
    """
    Wrapper function that uses the fast implementation
    """
    # Get statevector from circuit
    current_statevector = get_statevector(current_circuit)
    return compute_exact_commutator_gradient_with_statevector(current_statevector, H_qubit_op, generator_op, fragment_indices, n_qubits, shots_per_fragment)


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
def scipy_energy_optimization(H_sparse_matrix, current_state, new_operator,
                            initial_guess=0.01, max_iter=50, tol=1e-6):
    """
    Energy optimization using scipy.minimize for robust parameter optimization.

    Args:
        H_sparse_matrix: Sparse Hamiltonian matrix
        current_state: Current state vector
        new_operator: New operator to optimize
        initial_guess: Initial parameter guess
        max_iter: Maximum optimization iterations
        tol: Convergence tolerance

    Returns:
        optimal_param: Optimal parameter value
        optimal_energy: Optimal energy value
        final_state: Final state vector
    """

    def energy_function(param):
        energy, _ = fast_energy_calculation(H_sparse_matrix, current_state, new_operator, param[0])
        # Clean up after each energy calculation
        gc.collect()
        return energy

    # Use scipy.minimize with L-BFGS-B method for single parameter optimization
    from scipy.optimize import minimize

    # Try different optimization methods for robustness
    methods_to_try = ['L-BFGS-B', 'BFGS', 'CG']
    best_result = None
    best_energy = float('inf')

    for method in methods_to_try:
        try:
            result = minimize(
                energy_function,
                [initial_guess],
                method=method,
                options={
                    'maxiter': max_iter,
                    'disp': False,
                    'gtol': tol,
                    'ftol': tol
                },
                bounds=[(-np.pi, np.pi)] if method == 'L-BFGS-B' else None
            )

            if result.success and result.fun < best_energy:
                best_result = result
                best_energy = result.fun

        except Exception as e:
            print(f"    Method {method} failed: {e}")
            continue

    # If all methods failed, use the last result or fallback
    if best_result is None:
        print("    All optimization methods failed, using fallback")
        # Fallback to simple grid search
        param_range = np.linspace(-np.pi, np.pi, 100)
        energies = []
        for param in param_range:
            energy = energy_function([param])
            energies.append(energy)

        best_idx = np.argmin(energies)
        optimal_param = param_range[best_idx]
        optimal_energy = energies[best_idx]
    else:
        optimal_param = best_result.x[0]
        optimal_energy = best_result.fun

    # Calculate final state with optimal parameter
    optimal_energy, final_state = fast_energy_calculation(H_sparse_matrix, current_state, new_operator, optimal_param)

    # Aggressive cleanup
    del energy_function, best_result, best_energy
    gc.collect()
    print_memory_usage("after scipy_energy_optimization cleanup")

    return optimal_param, optimal_energy, final_state


def compute_exact_commutator_gradient_fast(current_circuit, H_qubit_op, generator_op, fragment_indices, n_qubits, shots_per_fragment=1024):
    """
    Ultra-fast computation of energy gradient using sparse matrix operations and optimized Pauli calculations
    """
    # Get the current statevector as numpy array for faster operations
    current_statevector = get_statevector(current_circuit)
    state_vector = current_statevector.data  # Convert to numpy array

    epsilons = [0.001, 0.01, 0.1]

    # Convert to qubit operator
    commutator_qubit = get_commutator_qubit(H_qubit_op, generator_op)

    # Decompose into QWC groups
    from utils.decomposition import qwc_decomposition
    pauli_groups = qwc_decomposition(commutator_qubit)

    fragment_expectations = []
    fragment_variances = []
    gradient_variance = 0

    for i, group in enumerate(pauli_groups):
        group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)

        # Vectorized computation for all Pauli strings in this group
        pauli_strings, coeffs = zip(*group_op.to_list())
        coeffs = np.array(coeffs)

        # Compute expectations for all Pauli strings in this group at once
        expectations = np.zeros(len(pauli_strings))

        for j, (pauli_string, coeff) in enumerate(zip(pauli_strings, coeffs)):
            # Use optimized Pauli expectation calculation
            expectation = compute_pauli_expectation_fast(state_vector, pauli_string, n_qubits)
            expectations[j] = expectation

        # Vectorized fragment computation
        fragment_exact_expectation = np.sum(coeffs * expectations)
        fragment_exact_variance = np.sum((coeffs**2) * (1.0 - expectations**2))

        gradient_variance += fragment_exact_variance

        # Calculate fragment statistics
        fragment_mean = fragment_exact_expectation
        fragment_variance = fragment_exact_variance

        fragment_expectations.append(fragment_mean)
        fragment_variances.append(fragment_variance)

    # Vectorized N_est calculation
    N_est = [gradient_variance / epsilon**2 for epsilon in epsilons]

    # Estimate total gradient and variance
    exact_gradient = sum(fragment_expectations)
    exact_variance = sum(fragment_variances)

    # Debug: print some information about the computation
    if abs(exact_gradient) > 1e-10:
        print(f"    Non-zero gradient found: {exact_gradient:.6e}, variance: {exact_variance:.6e}")
        print(f"    Number of fragments: {len(fragment_expectations)}")

    # Clean up large objects
    del current_statevector, commutator_qubit, pauli_groups, fragment_expectations, fragment_variances

    return exact_gradient, exact_variance, N_est


def compute_pauli_expectation_fast(state_vector, pauli_string, n_qubits):
    """
    Fast Pauli expectation calculation using Qiskit's optimized approach
    """
    # Use Qiskit's SparsePauliOp for reliable and fast computation
    pauli_op = SparsePauliOp.from_list([(pauli_string, 1.0)])

    # Convert state vector to Qiskit Statevector for compatibility
    from qiskit.quantum_info import Statevector
    qiskit_state = Statevector(state_vector)

    # Compute expectation value using Qiskit's optimized method
    expectation = qiskit_state.expectation_value(pauli_op)

    return np.real(expectation)


def _compute_single_exact_gradient_bai_with_statevector(args):
    """Worker function for parallel exact gradient computation in BAI using statevector directly"""
    try:
        arm_index, current_statevector, H_qubit_op, generator_op, fragment_indices, n_qubits, radius = args

        # Compute exact gradient for this arm using statevector directly
        estimate_gradient, estimate_variance, N_est, total_shots = compute_exact_commutator_gradient_with_statevector(
            current_statevector, H_qubit_op, generator_op, fragment_indices, n_qubits, radius=radius
        )

        # Clean up and return
        del current_statevector, H_qubit_op, generator_op, fragment_indices
        gc.collect()
        return arm_index, estimate_gradient, estimate_variance, N_est, total_shots

    except Exception as e:
        print(f"Error computing exact gradient for arm {arm_index}: {e}")
        gc.collect()
        return arm_index, 0.0, 1.0, 0.0, 0

def _compute_single_exact_gradient_bai(args):
    """Worker function for parallel exact gradient computation in BAI"""
    try:
        arm_index, current_circuit, H_qubit_op, generator_op, fragment_indices, n_qubits, shot_per_fragment = args

        # Compute exact gradient for this arm
        estimate_gradient, estimate_variance, N_est, total_shots = compute_exact_commutator_gradient(
            current_circuit, H_qubit_op, generator_op, fragment_indices, n_qubits, shots_per_fragment=shot_per_fragment
        )

        # Clean up and return
        del current_circuit, H_qubit_op, generator_op, fragment_indices
        gc.collect()
        return arm_index, estimate_gradient, estimate_variance, N_est, total_shots

    except Exception as e:
        print(f"Error computing exact gradient for arm {arm_index}: {e}")
        gc.collect()
        return arm_index, 0.0, 1.0


def bai_find_the_best_arm_exact_with_statevector(current_statevector, H_qubit_op, generator_pool, fragment_group_indices_map, commutator_indices_map, iteration, n_qubits, delta=0.05, max_rounds=10, shots_per_round=4096):
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
    rounds += 1
    print(f"Round {rounds}")

    args_list = [
        (i, current_statevector, H_qubit_op, generator_pool[i],
         commutator_indices_map[i], n_qubits, 1)
        for i in active_arms
    ]

    num_processes = min(6, len(active_arms))
    shots_across_gradient = 0

    try:
        with Pool(num_processes) as p:
            exact_results = p.map(
                _compute_single_exact_gradient_bai_with_statevector,
                args_list)

        # Store exact gradients and variances
        for arm_index, estimate_gradient, estimate_variance, N_est, total_shots in exact_results:
            estimated_gradients[arm_index] = estimate_gradient
            estimated_variances[arm_index] = estimate_variance
            exact_N_est += np.array(N_est, np.float64)
            shots_across_gradient += total_shots

        # Clean up parallel processing results
        del exact_results, args_list

    except Exception as e:
        print(
            f"Parallel exact gradient computation failed ({e}), falling back to sequential")
        # Fallback to sequential processing
        for i in active_arms:
            exact_grad, variance, N_est, total_shots = compute_exact_commutator_gradient_with_statevector(
                current_statevector, H_qubit_op, generator_pool[i],
                commutator_indices_map[i], n_qubits,
                1
            )
            estimated_gradients[i] = exact_grad
            estimated_variances[i] = variance
            exact_N_est += np.array(N_est, np.float64)

            # Clean up after each gradient computation
            gc.collect()

    total_measurements_across_fragments += shots_across_gradient
    measurements_trend_bai.append(shots_across_gradient)
    # Select best arm based on sampled estimates
    means = estimated_gradients
    best_arm = max(np.array(active_arms), key=lambda i: abs(means[i]))
    best_gradient = abs(means[best_arm])

    print(f"Final BAI result: {len(active_arms)} active arms remaining after {rounds} rounds")
    print(f"Selected best arm {best_arm} with sampled gradient magnitude {best_gradient:.6e}")

    return best_gradient, best_arm, total_measurements_across_fragments, measurements_trend_bai, exact_N_est

def bai_find_the_best_arm_exact(current_circuit, H_qubit_op, generator_pool, fragment_group_indices_map, commutator_indices_map, iteration, n_qubits, delta=0.05, max_rounds=10, shots_per_round=4096):
    """
    BAI algorithm using exact gradients as means and sampling from normal distributions.
    Computes exact gradients once, then samples from normal distributions for BAI rounds.
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
    active_qwc_groups = set(fragment_group_indices_map.values())
    rounds = 0
    total_measurements_across_fragments = 0
    measurements_trend_bai = []

    print(f"Number of active QWC groups: {len(active_qwc_groups)}")

    # First, compute exact gradients for all arms in parallel
    # These will be used as means for normal distribution sampling
    print("Computing exact gradients for all arms...")

    args_list = [
        (i, current_circuit, H_qubit_op, generator_pool[i], commutator_indices_map[i], n_qubits, shots_per_round)
        for i in active_arms
    ]

    num_processes = min(cpu_count() - 4, len(active_arms))

    try:
        with Pool(num_processes) as p:
            exact_results = p.map(_compute_single_exact_gradient_bai, args_list)

        # Store exact gradients and variances
        for arm_index, estimate_gradient, estimate_variance, N_est, total_shots in exact_results:
            estimated_gradients[arm_index] = estimate_gradient
            estimated_variances[arm_index] = estimate_variance
            exact_N_est += np.array(N_est, np.float64)

        print("Exact gradient computation completed")
        print(f"Sample estimate gradients: {[f'{estimated_gradients[i]:.6e}' for i in range(min(5, K))]}")
        print(f"Sample estimate variances: {[f'{estimated_variances[i]:.6e}' for i in range(min(5, K))]}")

        # Clean up parallel processing results
        del exact_results, args_list

    except Exception as e:
        print(f"Parallel exact gradient computation failed ({e}), falling back to sequential")
        # Fallback to sequential processing
        for i in active_arms:
            exact_grad, variance, N_est, total_shots = compute_exact_commutator_gradient(
                current_circuit, H_qubit_op, generator_pool[i],
                commutator_indices_map[i], n_qubits, shots_per_fragment=shots_per_round
            )
            estimated_gradients[i] = exact_grad
            estimated_variances[i] = variance
            exact_N_est += np.array(N_est, np.float64)

            # Clean up after each gradient computation
            gc.collect()

    best_arm = max(np.array(active_arms), key=lambda i: abs(estimated_gradients[i]))
    best_gradient = abs(estimated_gradients[best_arm])

    print(f"Final BAI result: {len(active_arms)} active arms remaining after {rounds} rounds")
    print(f"Selected best arm {best_arm} with sampled gradient magnitude {best_gradient:.6e}")

    return best_gradient, best_arm, total_measurements_across_fragments, measurements_trend_bai, exact_N_est


def get_counts_for_each_fragment_exact(current_circuit, fragment_group_indices_map, active_qwc_groups, n_qubits, shots=8192):
    """
    Generate counts for each fragment using normal distributions around exact gradient values
    with variances calculated from statevector diagonalization
    """
    # Get the current statevector
    current_statevector = get_statevector(current_circuit)

    counts_for_each_fragment = {}

    # For each active QWC group, compute exact expectation and variance
    for pauli_string, fragment_index in fragment_group_indices_map.items():
        if fragment_index in active_qwc_groups:
            # Create the Pauli operator
            pauli_op = SparsePauliOp.from_list([(pauli_string, 1.0)])

            # Calculate exact expectation value
            exact_expectation = current_statevector.expectation_value(pauli_op)

            # Calculate variance: Var(P) = 1 - ⟨P⟩²
            exact_variance = 1.0 - exact_expectation**2

            # Generate samples from normal distribution
            # Mean = exact expectation, Standard deviation = sqrt(variance)
            mean = exact_expectation
            std = np.sqrt(exact_variance)

            # Sample from normal distribution
            samples = np.random.normal(mean, std, shots)

            # Convert samples to counts (simulate measurement outcomes)
            # For Pauli measurements, outcomes are ±1
            # Convert continuous samples to discrete ±1 outcomes
            outcomes = np.where(samples >= 0, 1, -1)

            # Count occurrences
            unique, counts = np.unique(outcomes, return_counts=True)
            count_dict = dict(zip(unique, counts))

            # Convert to bitstring format expected by the rest of the code
            # '0' represents outcome +1, '1' represents outcome -1
            bitstring_counts = {}
            for outcome, count in count_dict.items():
                if outcome == 1:
                    bitstring_counts['0' * len(pauli_string)] = count
                else:  # outcome == -1
                    bitstring_counts['1' * len(pauli_string)] = count

            counts_for_each_fragment[fragment_index] = bitstring_counts

    return counts_for_each_fragment


def get_counts_for_each_fragment(current_circuit, fragment_group_indices_map, active_qwc_groups, n_qubits, shots=8192):
    """
    Wrapper function that uses the exact normal distribution approach
    """
    return get_counts_for_each_fragment_exact(current_circuit, fragment_group_indices_map, active_qwc_groups, n_qubits, shots)


def adapt_vqe_qiskit(H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op, generator_pool, fragment_group_indices_map, commutator_indices_map, shots=8192, max_iter=30, grad_tol=1e-4, verbose=True, mol='h4', save_intermediate=True, intermediate_filename='adapt_vqe_intermediate_results.csv', exact_energy=None):
    """
    ADAPT-VQE algorithm with multiprocessing support for gradient computation.

    Returns:
        tuple: (energies, params, ansatz_ops, final_state, total_measurements)
    """

    # Prepare reference state
    ansatz_ops = []
    params = []
    energies = []
    total_measurements = 0
    total_measurements_at_each_step = []
    total_measurements_trend_bai = {}

    # Compute exact energy if not provided
    if exact_energy is None:
        H_sparse = H_sparse_pauli_op.to_matrix(sparse=True)
        exact_energy, _ = exact_ground_state_energy(H_sparse)

    final_statevector = create_ansatz_statevector(n_qubits, n_electrons, ansatz_ops,
                                                  params, mol=mol)
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
        else:
            # For subsequent iterations, use the existing statevector directly
            current_statevector = final_statevector

        # Compute gradients for all pool operators using commutator measurement
        grads = []

        max_grad, best_idx, total_measurements_across_fragments, measurements_trend_bai, N_est = (
            bai_find_the_best_arm_exact_with_statevector(current_statevector, H_qubit_op, generator_pool, fragment_group_indices_map, commutator_indices_map, iteration, n_qubits, shots_per_round=int(shots)))

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
        ansatz_ops.append(qubit_operator_to_qiskit_operator(generator_pool[best_idx], n_qubits))
        params.append(0.0)

        # Replace the expensive optimization section with scipy.minimize optimization
        print(f"  Starting scipy.minimize optimization...")

        # Keep Hamiltonian sparse for memory efficiency
        H_sparse_matrix = H_sparse_pauli_op.to_matrix(sparse=True)

        # Get current state as numpy array for faster operations
        current_state = final_statevector.data

        # Use scipy.minimize optimization
        optimal_param, optimal_energy, updated_state = scipy_energy_optimization(
            H_sparse_matrix, current_state, ansatz_ops[-1],
            initial_guess=0.01, max_iter=30, tol=1e-6
        )

        # Update parameter and energy
        params[-1] = optimal_param
        energy = optimal_energy

        # Update final statevector
        final_statevector = Statevector(updated_state)
        energies.append(energy)

        if verbose:
            print(f"  Scipy.minimize optimization completed")
            print(f"  Energy after iteration {iteration}: {energy:.8f}")
            print(f"  Optimal parameter: {optimal_param:.6f}")

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
                filename=intermediate_filename
            )

        # Clean up memory
        del H_sparse_matrix, current_state, updated_state
        gc.collect()

    # Return final state using fast statevector approach
    final_state = create_ansatz_statevector(n_qubits, n_electrons, ansatz_ops, params, mol=mol)

    if verbose:
        print(f"Total measurements used: {total_measurements}")

    return energies, params, ansatz_ops, final_state, total_measurements, total_measurements_at_each_step, total_measurements_trend_bai

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python my_script.py <mol_file> <mol> <n_qubits> <n_electrons> <pool_type>")
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

    now = datetime.now()

    # Format as "YYYY-MM-DD HH:MM"
    time_string = now.strftime("%Y-%m-%d%H-%M")

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

    # Generate cache filename for commutator maps
    cache_filename = generate_commutator_cache_filename(mol, n_qubits, n_electrons, len(generator_pool), pool_type)

    # Get commutator maps with caching
    fragment_group_indices_map, commutator_indices_map = get_commutator_maps_cached(
        H_qubit_op, generator_pool, n_qubits,
        molecule_name=mol, n_electrons=n_electrons, cache_file=cache_filename)

    # print(f"Fragment Group Indices map: {fragment_group_indices_map}")
    #
    # exit()

    print(f"QWC groups found: {len(fragment_group_indices_map)}")
    print(f"Commutator mappings for {len(commutator_indices_map)} operators")


    # Configuration parameters
    use_parallel = True
    max_workers = None
    executor_type = 'multiprocessing'
    molecule_name = mol

    # Intermediate saving configuration
    save_intermediate = True
    intermediate_filename = f'adapt_vqe_intermediate_{mol}_{pool_type}_results_{time_string}_exact.csv'

    energies, params, ansatz, final_state, total_measurements, total_measurements_at_each_step, total_measurements_trend_bai = adapt_vqe_qiskit(
        H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op, generator_pool,
        fragment_group_indices_map, commutator_indices_map, mol=mol, exact_energy=exact_energy,
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
        filename=f'adapt_vqe_qubitwise_bai_{mol}_{pool_type}_results_exact.csv'
    )
