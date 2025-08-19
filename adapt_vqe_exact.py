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
from utils.reference_state_utils import get_reference_state, get_occ_no, get_bk_basis_states, find_index
from utils.ferm_utils import ferm_to_qubit
import pickle
import os
import sys
from get_generator_pool import get_generator_pool
from datetime import datetime

from qiskit.quantum_info import Statevector
from validations.hf_state_validation import validate_hartree_fock_state
from adapt_vqe_exact_bai_scipy_minimization_multi_params import scipy_multi_parameter_energy_optimization
from openfermion.linalg import qubit_operator_sparse
from sparse_energy_calculation import (sparse_multi_parameter_energy_optimization,
                                     sparse_single_parameter_energy_optimization,
                                     sparse_energy_calculation)
from openfermion import get_sparse_operator


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


def compute_exact_commutator_gradient_with_statevector(current_statevector, H_qubit_op, generator_op, n_qubits, radius):
    """
    Ultra-fast computation of energy gradient using sparse matrix operations and optimized Pauli calculations
    Uses statevector directly instead of circuit.
    """

    # Handle both Statevector objects and numpy arrays
    if hasattr(current_statevector, 'data'):
        state_vector = current_statevector.data  # Convert to numpy array
    else:
        state_vector = current_statevector  # Already a numpy array

        # Convert memoryview back to numpy array if needed (multiprocessing issue)
    if isinstance(state_vector, memoryview):
        state_vector = np.array(state_vector, dtype=np.complex128)
    elif not isinstance(state_vector, np.ndarray):
        state_vector = np.array(state_vector, dtype=np.complex128)
    elif state_vector.dtype != np.complex128:
        state_vector = state_vector.astype(np.complex128)

    epsilons = [radius, 0.01, 0.1]

    # Convert to qubit operator
    commutator_qubit = get_commutator_qubit(H_qubit_op, generator_op)

    # Convert to matrix
    commutator_matrix = get_sparse_operator(commutator_qubit, n_qubits)

    # Calculate expectation value directly: ⟨ψ|M|ψ⟩ = ψ† M ψ
    # Convert to dense matrix if needed
    if hasattr(commutator_matrix, 'toarray'):
        commutator_dense = commutator_matrix.toarray()
    else:
        commutator_dense = commutator_matrix

    # Compute M|ψ⟩ first, then ⟨ψ|(M|ψ⟩)
    commutator_applied = commutator_dense @ state_vector
    gradient = np.real(np.vdot(state_vector, commutator_applied))

    # Decompose into QWC groups
    from utils.decomposition import qwc_decomposition
    pauli_groups = qwc_decomposition(commutator_qubit)

    fragment_expectations = []
    fragment_variances = []
    gradient_variance = 0
    total_shots = 0

    # Initialize variables that might be used in cleanup
    expectations = None
    coeffs = None
    pauli_strings = None

    try:
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

            gradient_variance += fragment_exact_variance

            # Calculate fragment statistics
            fragment_mean = fragment_exact_expectation
            fragment_variance = fragment_exact_variance

            fragment_expectations.append(fragment_mean)
            fragment_variances.append(fragment_variance)
            total_shots += shots_per_fragment

    except Exception as e:
        print(f"    Error in gradient computation: {e}")
        # Return safe default values
        return 0.0, 1.0, [0.0, 0.0, 0.0], 0

    # Vectorized N_est calculation
    N_est = [gradient_variance / epsilon**2 for epsilon in epsilons]

    # Estimate total gradient and variance
    estimated_gradient = sum(fragment_expectations)
    estimated_variance = sum(fragment_variances)

    assert np.isclose(estimated_gradient, gradient), "Error in gradient computation"
    print(f"QWC expectation: {estimated_gradient}")
    print(f"Exact expectation: {gradient}")

    # Debug: print some information about the computation
    if abs(estimated_gradient) > 1e-10:
        print(f"    Non-zero gradient found: {estimated_gradient:.6e}, variance: {estimated_variance:.6e}")
        print(f"    Number of fragments: {len(fragment_expectations)}")

    # Clean up large objects - only delete if they exist
    try:
        del current_statevector, commutator_qubit, pauli_groups, fragment_expectations, fragment_variances
        del state_vector
        if expectations is not None:
            del expectations
        if coeffs is not None:
            del coeffs
        if pauli_strings is not None:
            del pauli_strings
    except NameError:
        pass  # Some variables might not exist
    gc.collect()

    return estimated_gradient, estimated_variance, N_est, total_shots

def compute_pauli_expectation_fast(state_vector, pauli_string, n_qubits):
    """
    Fast Pauli expectation calculation using Qiskit's optimized approach
    """
    try:
        # Use Qiskit's SparsePauliOp for reliable and fast computation
        pauli_op = SparsePauliOp.from_list([(pauli_string, 1.0)])

        # Ensure state_vector is in the right format for Qiskit
        if hasattr(state_vector, 'dtype') and state_vector.dtype == np.complex128:
            # Already a numpy array with complex dtype
            qiskit_state = Statevector(state_vector)
        elif hasattr(state_vector, 'dtype') and state_vector.dtype == np.float64:
            # Convert float64 to complex128
            state_vector_complex = state_vector.astype(np.complex128)
            qiskit_state = Statevector(state_vector_complex)
        else:
            # Convert to numpy array and ensure complex dtype
            state_vector_array = np.array(state_vector, dtype=np.complex128)
            qiskit_state = Statevector(state_vector_array)

        # Compute expectation value using Qiskit's optimized method
        expectation = qiskit_state.expectation_value(pauli_op)

        return np.real(expectation)

    except Exception as e:
        print(f"    Error in Pauli expectation calculation: {e}")
        print(f"    State vector type: {type(state_vector)}")
        if hasattr(state_vector, 'dtype'):
            print(f"    State vector dtype: {state_vector.dtype}")
        if hasattr(state_vector, 'shape'):
            print(f"    State vector shape: {state_vector.shape}")
        # Return a safe default value
        return 0.0

def compute_exact_commutator_gradient(current_circuit, H_qubit_op, generator_op, n_qubits, shots_per_fragment=1024):
    """
    Wrapper function that uses the fast implementation
    """
    # Get statevector from circuit
    current_statevector = get_statevector(current_circuit)
    return compute_exact_commutator_gradient_with_statevector(current_statevector, H_qubit_op, generator_op, n_qubits, shots_per_fragment)





def _compute_single_exact_gradient_bai_with_statevector(args):
    """Worker function for parallel exact gradient computation in BAI using statevector directly"""
    try:
        import numpy as np  # Import numpy in worker function
        arm_index, current_statevector, H_qubit_op, generator_op, n_qubits, radius = args

        # Handle both Statevector objects and numpy arrays
        if hasattr(current_statevector, 'data'):
            state_vector = current_statevector.data
        else:
            state_vector = current_statevector

        # Convert memoryview back to numpy array if needed (multiprocessing issue)
        if isinstance(state_vector, memoryview):
            state_vector = np.array(state_vector, dtype=np.complex128)
        elif not isinstance(state_vector, np.ndarray):
            state_vector = np.array(state_vector, dtype=np.complex128)
        elif state_vector.dtype != np.complex128:
            state_vector = state_vector.astype(np.complex128)

        # Compute exact gradient for this arm using statevector directly
        estimate_gradient, estimate_variance, N_est, total_shots = compute_exact_commutator_gradient_with_statevector(
            state_vector, H_qubit_op, generator_op, n_qubits, radius=radius
        )

        # Clean up and return
        del current_statevector, H_qubit_op, generator_op
        gc.collect()
        return arm_index, estimate_gradient, estimate_variance, N_est, total_shots

    except Exception as e:
        print(f"Error computing exact gradient for arm {arm_index}: {e}")
        # Print more detailed error information for debugging
        import traceback
        print(f"Full traceback for arm {arm_index}:")
        traceback.print_exc()
        gc.collect()
        return arm_index, 0.0, 1.0, [0.0, 0.0, 0.0], 0

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


def bai_find_the_best_arm_exact_with_statevector(current_statevector, H_qubit_op, generator_pool, iteration, n_qubits, delta=0.05, max_rounds=10, shots_per_round=4096):
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

    # Ensure statevector is properly converted to numpy array for multiprocessing
    state_vector_data = current_statevector.data if hasattr(current_statevector, 'data') else current_statevector

    # Ensure the state vector is a proper numpy array with complex dtype
    if not isinstance(state_vector_data, np.ndarray):
        state_vector_data = np.array(state_vector_data, dtype=np.complex128)
    elif state_vector_data.dtype != np.complex128:
        state_vector_data = state_vector_data.astype(np.complex128)

    # Debug: print information about the state vector before multiprocessing
    print(f"    State vector type before multiprocessing: {type(state_vector_data)}")
    print(f"    State vector dtype: {state_vector_data.dtype}")
    print(f"    State vector shape: {state_vector_data.shape}")
    print(f"    State vector norm: {np.linalg.norm(state_vector_data):.6f}")

    args_list = [
        (i, state_vector_data, H_qubit_op, generator_pool[i],
         n_qubits, 1)
        for i in active_arms
    ]

    num_processes = min(6, len(active_arms))
    shots_across_gradient = 0

    try:
        # Try to use multiprocessing with better error handling
        # Use default context first, fallback to spawn if needed
        try:
            print(f"    Using default multiprocessing with {num_processes} processes")
            with Pool(num_processes, maxtasksperchild=1) as p:
                exact_results = p.map(
                    _compute_single_exact_gradient_bai_with_statevector,
                    args_list)
        except Exception as mp_error:
            print(f"    Default multiprocessing failed ({mp_error}), trying spawn context")
            # Use 'spawn' context to avoid issues with certain objects
            ctx = get_context('spawn')
            with ctx.Pool(num_processes, maxtasksperchild=1) as p:
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
            try:
                exact_grad, variance, N_est, total_shots = compute_exact_commutator_gradient_with_statevector(
                    current_statevector, H_qubit_op, generator_pool[i],
                    n_qubits,
                    1
                )
                estimated_gradients[i] = exact_grad
                estimated_variances[i] = variance
                exact_N_est += np.array(N_est, np.float64)

                # Clean up after each gradient computation
                gc.collect()
            except Exception as inner_e:
                print(f"    Sequential gradient computation failed for arm {i}: {inner_e}")
                estimated_gradients[i] = 0.0
                estimated_variances[i] = 1.0
                exact_N_est += np.array([0.0, 0.0, 0.0], np.float64)

    total_measurements_across_fragments += shots_across_gradient
    measurements_trend_bai.append(shots_across_gradient)
    # Select best arm based on sampled estimates
    means = estimated_gradients
    best_arm = max(np.array(active_arms), key=lambda i: abs(means[i]))
    best_gradient = abs(means[best_arm])

    print(f"Final BAI result: {len(active_arms)} active arms remaining after {rounds} rounds")
    print(f"Selected best arm {best_arm} with sampled gradient magnitude {best_gradient:.6e}")

    return best_gradient, best_arm, total_measurements_across_fragments, measurements_trend_bai, exact_N_est

def bai_find_the_best_arm_exact(current_circuit, H_qubit_op, generator_pool, iteration, n_qubits, delta=0.05, max_rounds=10, shots_per_round=4096):
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
    rounds = 0
    total_measurements_across_fragments = 0
    measurements_trend_bai = []


    # First, compute exact gradients for all arms in parallel
    # These will be used as means for normal distribution sampling
    print("Computing exact gradients for all arms...")

    args_list = [
        (i, current_circuit, H_qubit_op, generator_pool[i], n_qubits, shots_per_round)
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
                n_qubits, shots_per_fragment=shots_per_round
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


def adapt_vqe_qiskit(H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op, generator_pool, shots=8192, max_iter=30, grad_tol=1e-4, verbose=True, mol='h4', save_intermediate=True, intermediate_filename='adapt_vqe_intermediate_results.csv', exact_energy=None):
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

    # Preparing the HF state and calculate the HF energy
    initial_state = create_ansatz_statevector(n_qubits, n_electrons,
                                              [],
                                              [], mol=mol)
    final_statevector = initial_state
    energy = measure_expectation_statevector(initial_state, H_sparse_pauli_op)
    print(f"HF energy (Qiskit): {energy}")

    for iteration in range(max_iter):
        # Use existing statevector instead of reconstructing circuit
        print(f'Iteration {iteration}, Length of Ansatz: {len(ansatz_ops)}, Parameters: {params}')

        # For the first iteration, we need to create the circuit to get the initial statevector
        # final_statevector: The evolved statevector from HF by adding the generators
        if iteration == 0:
            # Get the initial statevector
            current_statevector = create_ansatz_statevector(n_qubits, n_electrons, ansatz_ops,
                                                  params, mol=mol)
        else:
            # For subsequent iterations, use the existing statevector directly
            current_statevector = final_statevector

        # Compute gradients for all pool operators using commutator measurement
        grads = []

        max_grad, best_idx, total_measurements_across_fragments, measurements_trend_bai, N_est = (
            bai_find_the_best_arm_exact_with_statevector(current_statevector, H_qubit_op, generator_pool, iteration, n_qubits, shots_per_round=int(shots)))

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
        ansatz_ops.append(qubit_operator_sparse(generator_pool[best_idx], n_qubits))
        params.append(0.0)

        # Replace the expensive optimization section with scipy.minimize optimization
        print(f"  Starting scipy.minimize optimization...")

        # Keep Hamiltonian sparse for memory efficiency
        H_sparse_matrix = H_sparse_pauli_op.to_matrix(sparse=True)

        # Get current state as numpy array for faster operations
        # current_state = final_statevector.data


        if len(ansatz_ops) > 0:
            # Use sparse multi-parameter optimization to optimize all parameters together
            optimal_params, optimal_energy, updated_state = sparse_multi_parameter_energy_optimization(
                H_sparse_matrix, initial_state, ansatz_ops, params,
                max_iter=50, tol=1e-8
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
        energies.append(energy)

        if verbose:
            print(f"  Scipy.minimize optimization completed")
            print(f"  Energy after iteration {iteration}: {energy:.8f}")
            # print(f"  Optimal parameter: {optimal_param:.6f}")

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
        del H_sparse_matrix, updated_state
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
        mol=mol, exact_energy=exact_energy,
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
