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


def compute_exact_commutator_gradient(current_circuit, H_qubit_op, generator_op, fragment_indices, n_qubits, shots_per_fragment=1024):
    """
    Compute the energy gradient of the generator by sampling individual fragments
    :param current_circuit: Current ansatz circuit
    :param H_qubit_op: Hamiltonian in qubit operator form
    :param generator_op: Generator operator
    :param fragment_indices: The QWC fragments indices of the decomposed [H, G]
    :param n_qubits: Number of qubits
    :param shots_per_fragment: Number of shots to use for each fragment measurement
    :return: estimated_gradient, variance_estimate
    """

    # Get the current statevector
    current_statevector = get_statevector(current_circuit)

    epsilons = [0.001, 0.01, 0.1]

    # Convert to qubit operator
    commutator_qubit = get_commutator_qubit(H_qubit_op, generator_op)

    # Decompose into QWC groups
    from utils.decomposition import qwc_decomposition
    pauli_groups = qwc_decomposition(commutator_qubit)

    fragment_expectations = []
    fragment_variances = []

    gradient_variance = 0
    N_est = []

    for i, group in enumerate(pauli_groups):
        group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)

        # Calculate exact fragment expectation and variance
        fragment_exact_expectation = 0.0
        fragment_exact_variance = 0.0

        for pauli_string, coeff in group_op.to_list():
            # Create the Pauli operator
            pauli_op = SparsePauliOp.from_list([(pauli_string, 1.0)])

            # Calculate exact expectation value
            exact_expectation = current_statevector.expectation_value(pauli_op)

            # Add to fragment expectation
            fragment_exact_expectation += coeff * exact_expectation

            # Add to fragment variance (variance of Pauli measurement is 1 - expectation^2)
            pauli_variance = 1.0 - exact_expectation**2
            fragment_exact_variance += (coeff**2) * pauli_variance

            # Clean up Pauli operator
            del pauli_op

        # Sample from normal distribution with exact expectation as mean and sqrt(variance) as std
        fragment_samples = np.random.normal(
            fragment_exact_expectation,
            np.sqrt(fragment_exact_variance),
            int(np.ceil(fragment_exact_variance / (0.001 ** 2)))
        )

        gradient_variance += fragment_exact_variance

        # Calculate fragment statistics
        fragment_mean = np.mean(fragment_samples)
        fragment_variance = np.var(fragment_samples, ddof=1)

        fragment_expectations.append(fragment_mean)
        fragment_variances.append(fragment_variance)

        # Clean up group operator and samples
        del group_op, fragment_samples

    for epsilon in epsilons:
        N_est.append(gradient_variance / epsilon ** 2)


    # Estimate total gradient as sum of fragment expectations
    estimated_gradient = sum(fragment_expectations)

    # Estimate total variance as sum of fragment variances
    estimated_variance = sum(fragment_variances)

    # Debug: print some information about the computation
    if abs(estimated_gradient) > 1e-10:
        print(f"    Non-zero gradient found: {estimated_gradient:.6e}, variance: {estimated_variance:.6e}")
        print(f"    Number of fragments: {len(fragment_expectations)}")

    # Clean up large objects
    del current_statevector, commutator_qubit, pauli_groups, fragment_expectations, fragment_variances

    return estimated_gradient, estimated_variance, N_est


def _compute_single_exact_gradient_bai(args):
    """Worker function for parallel exact gradient computation in BAI"""
    try:
        arm_index, current_circuit, H_qubit_op, generator_op, fragment_indices, n_qubits, shot_per_fragment = args

        # Compute exact gradient for this arm
        estimate_gradient, estimate_variance, N_est = compute_exact_commutator_gradient(
            current_circuit, H_qubit_op, generator_op, fragment_indices, n_qubits, shots_per_fragment=shot_per_fragment
        )

        # Clean up and return
        del current_circuit, H_qubit_op, generator_op, fragment_indices
        gc.collect()
        return arm_index, estimate_gradient, estimate_variance, N_est

    except Exception as e:
        print(f"Error computing exact gradient for arm {arm_index}: {e}")
        gc.collect()
        return arm_index, 0.0, 1.0


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
        for arm_index, estimate_gradient, estimate_variance, N_est in exact_results:
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
            exact_grad, variance, N_est = compute_exact_commutator_gradient(
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


def adapt_vqe_qiskit(H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op, generator_pool, fragment_group_indices_map, commutator_indices_map, shots=8192, max_iter=30, grad_tol=1e-2, verbose=True, mol='h4', save_intermediate=True, intermediate_filename='adapt_vqe_intermediate_results.csv', exact_energy=None):
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
        # Create current ansatz circuit
        print(f'Iteration {iteration}, Length of Ansatz: {len(ansatz_ops)}, Parameters: {params}')
        current_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops, params)

        # Compute gradients for all pool operators using commutator measurement
        grads = []



        max_grad, best_idx, total_measurements_across_fragments, measurements_trend_bai, N_est = (
            bai_find_the_best_arm_exact(current_circuit, H_qubit_op, generator_pool, fragment_group_indices_map, commutator_indices_map, iteration, n_qubits, shots_per_round=int(shots)))

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

                # Optimize parameters using scipy.optimize.minimize
                # Ultra-fast single parameter optimization (fix existing parameters)
        # This approach is much faster and more numerically stable than full re-optimization
        energy_cache = {}

        # For single parameter optimization, we only optimize the new parameter
        # while keeping all previous parameters fixed at their current values
        def vqe_obj_single(new_param):
            # Create parameter vector with existing params + new param
            full_params = params[:-1] + [new_param]  # All previous + new parameter

            # Convert to tuple for hashing
            param_tuple = tuple(full_params)
            if param_tuple in energy_cache:
                return energy_cache[param_tuple]

            # Create statevector directly (much faster than circuit construction)
            statevector = create_ansatz_statevector(n_qubits, n_electrons, ansatz_ops, full_params, mol=mol)
            energy = measure_expectation_statevector(statevector, H_sparse_pauli_op)
            energy_cache[param_tuple] = energy
            return energy


        # Try optimization with different methods and starting points
        initial_guess = params.copy()
        print(f"  Starting optimization from: {initial_guess}")

        # Use faster optimization with fewer function evaluations for large systems
        max_iter = 30 if len(ansatz_ops) > 5 else 50  # Reduced iterations


        # For small number of parameters, use single parameter optimization
        # Optimize only the new parameter while keeping others fixed
        res = minimize(lambda x: vqe_obj_single(x[0]), [0.01], method='L-BFGS-B',
                      options={'maxiter': max_iter, 'disp': False, 'gtol': 1e-4})
        params[-1] = res.x[0]  # Update only the new parameter

        # Update energy using fast statevector approach
        final_statevector = create_ansatz_statevector(n_qubits, n_electrons, ansatz_ops, params, mol=mol)
        energy = measure_expectation_statevector(final_statevector, H_sparse_pauli_op)
        energies.append(energy)

        if verbose:
            print(f"  Energy after iteration {iteration}: {energy:.8f}")
            if len(energy_cache) > 0:
                print(f"  Single parameter optimization completed in {len(energy_cache)} function evaluations")
            else:
                print(f"  Single parameter optimization completed with ultra-fast line search")

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
                filename=intermediate_filename
            )

        # Clean up memory after each iteration
        del energy_cache, final_statevector
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
    intermediate_filename = f'adapt_vqe_intermediate_{mol}_{pool_type}_results_{time_string}.csv'

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
        filename=f'adapt_vqe_qubitwise_bai_{mol}_{pool_type}_results.csv'
    )
