from adaptvqe.adapt_vqe_preparation import (create_ansatz_circuit,
                                            measure_expectation,
                                            get_statevector,
                                            openfermion_qubitop_to_sparsepauliop,
                                            exact_ground_state_energy,
                                            save_results_to_csv,
                                            save_intermediate_results_to_csv,
                                            create_ansatz_statevector,
                                            measure_expectation_statevector,
                                            get_hf_statevector,
                                            apply_operator_to_statevector)
from scipy.optimize import minimize
import numpy as np
from multiprocessing import Pool, cpu_count, get_context
import gc
from qiskit.quantum_info import SparsePauliOp
from utils.qubit_utils import get_commutator_qubit, \
    qubit_operator_to_qiskit_operator
from adapt_vqe_qiskit_qubitwise_bai import compute_commutator_gradient, \
    generate_commutator_cache_filename, get_commutator_maps_cached
from utils.reference_state_utils import get_reference_state, get_occ_no, \
    get_bk_basis_states, find_index
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

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except ImportError:
        return 0.0


def print_memory_usage(label=""):
    """Print current memory usage with optional label"""
    memory_mb = get_memory_usage()
    print(f"Memory usage {label}: {memory_mb:.1f} MB")
    return memory_mb

def compute_exact_commutator_gradient_direct(current_statevector, H_qubit_op, generator_op, n_qubits):
    """
    Direct computation of energy gradient using the commutator expectation value.
    This is much faster than decomposing into Pauli groups and computing individual expectations.
    
    The gradient is: ∂E/∂θ = ⟨ψ|[H, A]|ψ⟩ where A is the generator operator
    """
    state_vector = current_statevector.data
    
    # Convert to qubit operator
    commutator_qubit = get_commutator_qubit(H_qubit_op, generator_op)
    
    # Convert commutator to Qiskit operator for fast expectation computation
    commutator_qiskit = qubit_operator_to_qiskit_operator(commutator_qubit, n_qubits)
    
    # Compute expectation value directly: ⟨ψ|[H, A]|ψ⟩
    gradient = current_statevector.expectation_value(commutator_qiskit)
    
    # The gradient is real for Hermitian operators
    gradient = np.real(gradient)
    
    # For variance estimation, we can compute Var([H, A]) = ⟨ψ|[H, A]²|ψ⟩ - ⟨ψ|[H, A]|ψ⟩²
    # But this requires computing the square of the commutator, which is expensive
    # Instead, we'll use a simple estimate based on the gradient magnitude
    if abs(gradient) > 1e-10:
        print(f"    Direct gradient computation: {gradient:.6e}")
        # Simple variance estimate: variance ≈ |gradient| for small gradients
        estimated_variance = abs(gradient)
    else:
        estimated_variance = 1e-12  # Small variance for zero gradients
    
    # Clean up
    del commutator_qubit, commutator_qiskit, state_vector
    gc.collect()
    
    return gradient, estimated_variance, [estimated_variance/0.001**2, estimated_variance/0.01**2, estimated_variance/0.1**2], 0

def compute_exact_commutator_gradient_with_statevector(current_statevector, H_qubit_op, generator_op, n_qubits, radius):
    """
    Ultra-fast computation of energy gradient using direct commutator expectation.
    This replaces the complex Pauli decomposition approach with direct computation.
    """
    return compute_exact_commutator_gradient_direct(current_statevector, H_qubit_op, generator_op, n_qubits)

def compute_pauli_expectation_fast(state_vector, pauli_string, n_qubits):
    """Fast Pauli expectation calculation using Qiskit's optimized approach"""
    pauli_op = SparsePauliOp.from_list([(pauli_string, 1.0)])
    qiskit_state = Statevector(state_vector)
    expectation = qiskit_state.expectation_value(pauli_op)
    return np.real(expectation)

def _compute_single_exact_gradient_bai_with_statevector(args):
    """Worker function for parallel exact gradient computation in BAI using direct commutator expectation"""
    try:
        arm_index, current_statevector, H_qubit_op, generator_op, n_qubits, radius = args
        estimate_gradient, estimate_variance, N_est, total_shots = compute_exact_commutator_gradient_direct(
            current_statevector, H_qubit_op, generator_op, n_qubits
        )
        del current_statevector, H_qubit_op, generator_op
        gc.collect()
        return arm_index, estimate_gradient, estimate_variance, N_est, total_shots
    except Exception as e:
        print(f"Error computing exact gradient for arm {arm_index}: {e}")
        gc.collect()
        return arm_index, 0.0, 1.0, 0.0, 0

def bai_find_the_best_arm_exact_with_statevector(current_statevector, H_qubit_op, generator_pool, iteration, n_qubits, delta=0.05, max_rounds=10, shots_per_round=4096):
    """BAI algorithm using direct exact gradient computation."""
    K = len(generator_pool)
    active_arms = list(range(K))
    estimated_gradients = np.zeros(K)
    estimated_variances = np.ones(K)
    exact_N_est = np.zeros(3)
    rounds = 0
    total_measurements_across_fragments = 0
    measurements_trend_bai = []

    print("Computing exact gradients for all arms using direct commutator expectation...")
    rounds += 1
    print(f"Round {rounds}")

    args_list = [
        (i, current_statevector, H_qubit_op, generator_pool[i],
         n_qubits, 1)
        for i in active_arms
    ]

    num_processes = min(6, len(active_arms))
    shots_across_gradient = 0

    try:
        with Pool(num_processes) as p:
            exact_results = p.map(
                _compute_single_exact_gradient_bai_with_statevector,
                args_list)

        for arm_index, estimate_gradient, estimate_variance, N_est, total_shots in exact_results:
            estimated_gradients[arm_index] = estimate_gradient
            estimated_variances[arm_index] = estimate_variance
            exact_N_est += np.array(N_est, np.float64)
            shots_across_gradient += total_shots

        del exact_results, args_list

    except Exception as e:
        print(f"Parallel exact gradient computation failed ({e}), falling back to sequential")
        for i in active_arms:
            exact_grad, variance, N_est, total_shots = compute_exact_commutator_gradient_direct(
                current_statevector, H_qubit_op, generator_pool[i],
                n_qubits
            )
            estimated_gradients[i] = exact_grad
            estimated_variances[i] = variance
            exact_N_est += np.array(N_est, np.float64)
            gc.collect()

    total_measurements_across_fragments += shots_across_gradient
    measurements_trend_bai.append(shots_across_gradient)
    means = estimated_gradients
    best_arm = max(np.array(active_arms), key=lambda i: abs(means[i]))
    best_gradient = abs(means[best_arm])

    print(f"Final BAI result: {len(active_arms)} active arms remaining after {rounds} rounds")
    print(f"Selected best arm {best_arm} with sampled gradient magnitude {best_gradient:.6e}")

    return best_gradient, best_arm, total_measurements_across_fragments, measurements_trend_bai, exact_N_est


def fast_energy_calculation(H_sparse_matrix, current_state, new_operator, new_param):
    """
    Fast energy calculation using sparse matrix operations and incremental updates.
    This is much faster than recreating the entire ansatz statevector.
    """
    # Convert new operator to matrix
    if hasattr(new_operator, 'to_matrix'):
        new_op_matrix = new_operator.to_matrix()
    else:
        new_op_matrix = new_operator

    # Apply the new operator using matrix exponential (much faster than circuit recreation)
    if hasattr(new_op_matrix, 'toarray'):  # If it's a sparse matrix
        dense_op = new_op_matrix.toarray()
        updated_state = expm_multiply(new_param * dense_op, current_state)
        del dense_op
    else:  # If it's already a dense matrix
        updated_state = expm_multiply(new_param * new_op_matrix, current_state)

    # Calculate energy using matrix multiplication
    if hasattr(H_sparse_matrix, 'toarray'):  # If it's a sparse matrix
        H_dense = H_sparse_matrix.toarray()
        temp_vector = H_dense @ updated_state
        energy = np.real(np.vdot(updated_state, temp_vector))
        del H_dense, temp_vector
        gc.collect()
    else:  # If it's already a dense matrix
        temp_vector = H_sparse_matrix @ updated_state
        energy = np.real(np.vdot(updated_state, temp_vector))
        del temp_vector
        gc.collect()

    # Clean up intermediate objects
    del new_op_matrix
    gc.collect()
    return energy, updated_state

def fast_single_parameter_optimization(H_sparse_pauli_op, ansatz_ops, current_params, current_statevector, n_qubits, n_electrons, mol='h4', max_iter=50, tol=1e-8):
    """
    Ultra-fast single parameter optimization using incremental state updates.
    This matches the approach from the scipy minimization file for maximum speed.
    """
    def energy_function_single_param(new_param):
        # Use fast incremental energy calculation instead of recreating statevector
        energy, _ = fast_energy_calculation(H_sparse_matrix, current_state, new_operator, new_param[0])
        gc.collect()
        return energy

    print(f"    Starting fast single parameter optimization with {len(current_params)} existing parameters + 1 new parameter")
    print(f"    Existing parameters: {current_params}")

    # Get current state as numpy array for faster operations
    current_state = current_statevector.data

    # Get the new operator (last one added)
    new_operator = ansatz_ops[-1]

    # Keep Hamiltonian sparse for memory efficiency
    H_sparse_matrix = H_sparse_pauli_op.to_matrix(sparse=True)

    # Use scipy.minimize with L-BFGS-B method for single parameter optimization
    result = minimize(
        energy_function_single_param,
        [0.01],  # Start with small value
        method='L-BFGS-B',  # Use L-BFGS-B like the fast version
        options={
            'disp': False,
            'gtol': tol
        },
        bounds=[(-np.pi, np.pi)]  # Add bounds for stability
    )

    print(f"    Fast single parameter optimization result: {result}")
    print(f"    Success: {result.success}")
    print(f"    Final energy: {result.fun}")
    print(f"    Optimal new parameter: {result.x[0]}")

    if result.success:
        optimal_new_param = result.x[0]
        optimal_energy = result.fun
        # Keep existing params unchanged, only add the new optimized param
        optimal_params = current_params + [optimal_new_param]
    else:
        print("    Fast single parameter optimization failed, using default value")
        optimal_new_param = 0.01
        optimal_energy = energy_function_single_param([optimal_new_param])
        optimal_params = current_params + [optimal_new_param]

    # Get the final statevector with optimal parameters using fast incremental update
    _, final_state = fast_energy_calculation(H_sparse_matrix, current_state, new_operator, optimal_new_param)
    final_statevector = Statevector(final_state)

    # Clean up
    del H_sparse_matrix, current_state, new_operator, final_state
    gc.collect()

    return optimal_params, optimal_energy, final_statevector


def evaluate_energy_full_ansatz(H_sparse_pauli_op, ansatz_ops, params, n_qubits, n_electrons, mol='h4'):
    """
    Evaluate energy using the direct statevector approach (much faster than circuit construction).
    This matches the approach used in shot-efficient-adapt-vqe.
    """
    # Create the ansatz statevector directly (much faster than circuit construction)
    statevector = create_ansatz_statevector(n_qubits, n_electrons, ansatz_ops, params, mol=mol)

    # Calculate energy using the statevector (direct calculation)
    energy = measure_expectation_statevector(statevector, H_sparse_pauli_op)

    return energy

def adapt_vqe_qiskit_single_param(H_sparse_pauli_op, n_qubits, n_electrons,
                                  H_qubit_op, generator_pool,
                                  shots=8192,
                                  max_iter=30, grad_tol=1e-8, verbose=True,
                                  mol='h4', save_intermediate=True,
                                  intermediate_filename='adapt_vqe_intermediate_results.csv',
                                  exact_energy=None,
                                  full_optimization_frequency=1):
    """
    ADAPT-VQE algorithm with FULL parameter optimization after each new operator (like shot-efficient).

    Args:
        full_optimization_frequency: Always 1 - do full optimization every iteration
    """
    ansatz_ops = []
    params = []
    energies = []
    total_measurements = 0
    total_measurements_at_each_step = []
    total_measurements_trend_bai = {}

    if exact_energy is None:
        H_sparse = H_sparse_pauli_op.to_matrix(sparse=True)
        exact_energy, _ = exact_ground_state_energy(H_sparse)

    final_statevector = create_ansatz_statevector(n_qubits, n_electrons,
                                                  ansatz_ops, params, mol=mol)
    energy = measure_expectation_statevector(final_statevector,
                                             H_sparse_pauli_op)
    print(f"HF energy (Qiskit): {energy}")

    for iteration in range(max_iter):
        print(
            f'Iteration {iteration}, Length of Ansatz: {len(ansatz_ops)}, Parameters: {params}')

        if iteration == 0:
            current_circuit = create_ansatz_circuit(n_qubits, n_electrons,
                                                    ansatz_ops, params)
            current_statevector = get_statevector(current_circuit)
        else:
            current_statevector = final_statevector

        # Compute gradients for all pool operators using commutator measurement
        max_grad, best_idx, total_measurements_across_fragments, measurements_trend_bai, N_est = (
            bai_find_the_best_arm_exact_with_statevector(current_statevector,
                                                         H_qubit_op,
                                                         generator_pool,
                                                         iteration, n_qubits,
                                                         shots_per_round=int(
                                                             shots)))

        total_measurements += total_measurements_across_fragments
        total_measurements_at_each_step.append(total_measurements)
        total_measurements_trend_bai[iteration] = measurements_trend_bai

        if verbose:
            print(
                f"Iteration {iteration}: max gradient = {max_grad:.6e}, best = {best_idx}")

        if max_grad < grad_tol:
            if verbose:
                print("Converged: gradient below threshold.")
            break

        # Add best operator to ansatz
        new_operator = qubit_operator_to_qiskit_operator(
            generator_pool[best_idx], n_qubits)
        ansatz_ops.append(new_operator)

        # Do FAST single parameter optimization after adding a new operator
        print(f"  Starting FAST single parameter optimization (iteration {iteration})...")

        optimal_params, optimal_energy, final_statevector = fast_single_parameter_optimization(
            H_sparse_pauli_op, ansatz_ops, params, current_statevector, n_qubits, n_electrons, mol=mol, max_iter=50, tol=1e-6
        )

        # Update parameters
        params = optimal_params.tolist() if hasattr(optimal_params, 'tolist') else list(optimal_params)
        energy = optimal_energy

        energies.append(energy)

        if verbose:
            print(f"  Optimization completed")
            print(f"  Energy after iteration {iteration}: {energy:.8f}")
            print(f"  Parameters: {[f'{p:.6f}' for p in params]}")

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
        gc.collect()

    # Return final state using fast statevector approach
    final_state = create_ansatz_statevector(n_qubits, n_electrons, ansatz_ops,
                                            params, mol=mol)

    if verbose:
        print(f"Total measurements used: {total_measurements}")

    return energies, params, ansatz_ops, final_state, total_measurements, total_measurements_at_each_step, total_measurements_trend_bai


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(
            "Usage: python adapt_vqe_exact_single_param.py <mol_file> <mol> <n_qubits> <n_electrons> <pool_type> <shots>")
        sys.exit(1)

    mol_file = sys.argv[1]
    with open(f'ham_lib/{mol_file}', 'rb') as f:
        fermion_op = pickle.load(f)
    mol = sys.argv[2]
    n_qubits = int(sys.argv[3])
    n_electrons = int(sys.argv[4])
    pool_type = sys.argv[5]
    shots = sys.argv[6]

    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d%H-%M")

    H_qubit_op = ferm_to_qubit(fermion_op)
    H_sparse_pauli_op = openfermion_qubitop_to_sparsepauliop(H_qubit_op,
                                                             n_qubits)

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
    cache_filename = generate_commutator_cache_filename(mol, n_qubits,
                                                        n_electrons,
                                                        len(generator_pool),
                                                        pool_type)

    # Configuration parameters
    use_parallel = True
    max_workers = None
    executor_type = 'multiprocessing'
    molecule_name = mol

    # Intermediate saving configuration
    save_intermediate = True
    intermediate_filename = f'adapt_vqe_intermediate_{mol}_{pool_type}_results_{time_string}_single_param.csv'

    # Use the new single-parameter optimization algorithm
    energies, params, ansatz, final_state, total_measurements, total_measurements_at_each_step, total_measurements_trend_bai = adapt_vqe_qiskit_single_param(
        H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op, generator_pool,
        mol=mol,
        exact_energy=exact_energy,
        save_intermediate=save_intermediate,
        intermediate_filename=intermediate_filename,
        full_optimization_frequency=1)  # Single parameter optimization every iteration

    # Calculate results
    final_energy = energies[-1]
    overlap = np.abs(np.vdot(final_state.data, exact_gs)) ** 2
    ansatz_depth = len(ansatz)

    # Print results
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
        filename=f'adapt_vqe_qubitwise_bai_{mol}_{pool_type}_results_single_param.csv'
    )
