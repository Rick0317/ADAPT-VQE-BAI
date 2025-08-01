from adaptvqe.adapt_vqe_preparation import (create_ansatz_circuit, measure_expectation,
                              get_statevector, openfermion_qubitop_to_sparsepauliop,
                              exact_ground_state_energy, save_results_to_csv)
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


def compute_exact_commutator_gradient(current_circuit, H_qubit_op, generator_op, fragment_indices, n_qubits):
    """
    Compute the exact energy gradient of the generator using statevector simulation
    :param current_circuit: Current ansatz circuit
    :param H_qubit_op: Hamiltonian in qubit operator form
    :param generator_op: Generator operator
    :param fragment_indices: The QWC fragments indices of the decomposed [H, G]
    :param n_qubits: Number of qubits
    :return: exact_gradient, variance_estimate
    """

    # Get the current statevector
    current_statevector = get_statevector(current_circuit)

    # Convert to qubit operator
    commutator_qubit = get_commutator_qubit(H_qubit_op, generator_op)

    # Decompose into QWC groups
    from utils.decomposition import qwc_decomposition
    pauli_groups = qwc_decomposition(commutator_qubit)

    total_expectation = 0.0
    variance_components = []

    for i, group in enumerate(pauli_groups):
        group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)

        # Calculate exact expectation for this group
        group_expectation = 0.0
        group_variance = 0.0

        for pauli_string, coeff in group_op.to_list():
            # Create the Pauli operator
            pauli_op = SparsePauliOp.from_list([(pauli_string, 1.0)])

            # Calculate exact expectation value
            expectation = current_statevector.expectation_value(pauli_op)
            group_expectation += coeff * expectation

            # Calculate variance contribution (for variance estimation)
            # Variance of a Pauli measurement is 1 - expectation^2
            pauli_variance = 1.0 - expectation**2
            group_variance += (coeff**2) * pauli_variance

        total_expectation += group_expectation
        variance_components.append(group_variance)

    # Estimate total variance (sum of individual group variances)
    total_variance = sum(variance_components)

    # Debug: print some information about the computation
    if abs(total_expectation) > 1e-10:
        print(f"    Non-zero gradient found: {total_expectation:.6e}, variance: {total_variance:.6e}")

    return total_expectation, total_variance


def _compute_single_exact_gradient_bai(args):
    """Worker function for parallel exact gradient computation in BAI"""
    try:
        arm_index, current_circuit, H_qubit_op, generator_op, fragment_indices, n_qubits = args

        # Compute exact gradient for this arm
        exact_gradient, variance = compute_exact_commutator_gradient(
            current_circuit, H_qubit_op, generator_op, fragment_indices, n_qubits
        )

        # Clean up and return
        gc.collect()
        return arm_index, exact_gradient, variance

    except Exception as e:
        print(f"Error computing exact gradient for arm {arm_index}: {e}")
        gc.collect()
        return arm_index, 0.0, 1.0


def bai_find_the_best_arm_exact(current_circuit, H_qubit_op, generator_pool, fragment_group_indices_map, commutator_indices_map, iteration, n_qubits, delta=0.05, max_rounds=10, shots_per_round=512):
    """
    BAI algorithm using exact gradients as means and sampling from normal distributions
    """
    K = len(generator_pool)
    active_arms = list(range(K))

    # Store exact gradients and variances
    exact_gradients = np.zeros(K)
    gradient_variances = np.ones(K)  # Initialize with 1.0

    # Track empirical estimates and pulls for BAI algorithm
    estimates = np.zeros(K)
    pulls = np.zeros(K)

    # Get all QWC groups that need to be computed
    active_qwc_groups = set(fragment_group_indices_map.values())
    rounds = 0
    total_measurements_across_fragments = 0
    measurements_trend_bai = []

    print(f"Number of active QWC groups: {len(active_qwc_groups)}")

    # First, compute exact gradients for all arms in parallel
    print("Computing exact gradients for all arms...")

    args_list = [
        (i, current_circuit, H_qubit_op, generator_pool[i], commutator_indices_map[i], n_qubits)
        for i in active_arms
    ]

    num_processes = min(cpu_count() - 4, len(active_arms))

    try:
        with Pool(num_processes) as p:
            exact_results = p.map(_compute_single_exact_gradient_bai, args_list)

        # Store exact gradients and variances
        for arm_index, exact_grad, variance in exact_results:
            exact_gradients[arm_index] = exact_grad
            gradient_variances[arm_index] = variance

        print("Exact gradient computation completed")
        print(f"Sample exact gradients: {[f'{exact_gradients[i]:.6e}' for i in range(min(5, K))]}")
        print(f"Sample variances: {[f'{gradient_variances[i]:.6e}' for i in range(min(5, K))]}")

    except Exception as e:
        print(f"Parallel exact gradient computation failed ({e}), falling back to sequential")
        # Fallback to sequential processing
        for i in active_arms:
            exact_grad, variance = compute_exact_commutator_gradient(
                current_circuit, H_qubit_op, generator_pool[i],
                commutator_indices_map[i], n_qubits
            )
            exact_gradients[i] = exact_grad
            gradient_variances[i] = variance

    # Now run BAI algorithm using sampling from normal distributions
    while len(active_arms) > 1 and rounds < max_rounds:
        rounds += 1
        print(f"Round {rounds}")

        # Sample from normal distributions for active arms
        for i in active_arms:
            # Use exact gradient as mean and sqrt(variance) as standard deviation
            mean = exact_gradients[i]
            std = np.sqrt(max(gradient_variances[i], 1e-6))  # Minimum variance to avoid zero std

            # Sample from normal distribution
            samples = np.random.normal(mean, std, shots_per_round)

            # Debug: print sampling info for first few arms
            if i < 3:
                print(f"    Arm {i}: mean={mean:.6e}, std={std:.6e}, sample_mean={np.mean(samples):.6e}")

            # Update empirical estimates using exponential moving average
            if pulls[i] == 0:
                estimates[i] = np.mean(samples)
            else:
                # Weighted average: new_estimate = (old_estimate * old_pulls + new_sample) / (old_pulls + 1)
                new_samples_mean = np.mean(samples)
                estimates[i] = (estimates[i] * pulls[i] + new_samples_mean * shots_per_round) / (pulls[i] + shots_per_round)

            pulls[i] += shots_per_round

        total_measurements_across_fragments += shots_per_round * len(active_arms)
        measurements_trend_bai.append(total_measurements_across_fragments)

        # Use empirical estimates for BAI decision
        means = estimates
        max_mean = max(abs(means[active_arms]))

        # Calculate confidence intervals based on empirical variance
        # For normal distribution, radius = sqrt(variance / pulls) * confidence_factor
        confidence_factor = np.sqrt(2 * np.log(len(active_arms) / delta))

        # Use empirical variance (estimated from the normal distribution variance)
        empirical_variances = gradient_variances / pulls  # Variance of the mean
        radius = np.sqrt(empirical_variances) * confidence_factor

        # Eliminate arms based on confidence intervals
        new_active_arms = []
        for i in active_arms:
            if abs(means[i]) + radius[i] >= max_mean - radius[i]:
                new_active_arms.append(i)

        active_arms = new_active_arms

        print(f"After round {rounds}, active_arms: {active_arms}")
        empirical_estimates = [f"{means[i]:.6e}" for i in active_arms]
        exact_grads = [f"{exact_gradients[i]:.6e}" for i in active_arms]
        print(f"Empirical estimates: {empirical_estimates}")
        print(f"Exact gradients: {exact_grads}")
        gc.collect()

    # Select best arm based on empirical estimates
    means = estimates
    best_arm = max(np.array(active_arms), key=lambda i: abs(means[i]))
    best_gradient = abs(means[best_arm])

    print(f"Final BAI result: {len(active_arms)} active arms remaining after {rounds} rounds")
    print(f"Selected best arm {best_arm} with empirical gradient magnitude {best_gradient:.6e}")
    print(f"Exact gradient for best arm: {abs(exact_gradients[best_arm]):.6e}")

    return best_gradient, best_arm, total_measurements_across_fragments, measurements_trend_bai


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


def adapt_vqe_qiskit(H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op, generator_pool, fragment_group_indices_map, commutator_indices_map, max_iter=30, grad_tol=1e-2, verbose=True, use_multiprocessing=True):
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

    final_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops,
                                          params)
    energy = measure_expectation(final_circuit, H_sparse_pauli_op)
    print(f"HF energy (Qiskit): {energy}")

    for iteration in range(max_iter):
        # Create current ansatz circuit
        print(f'Iteration {iteration}, Length of Ansatz: {len(ansatz_ops)}, Parameters: {params}')
        current_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops, params)

        # Compute gradients for all pool operators using commutator measurement
        grads = []



        max_grad, best_idx, total_measurements_across_fragments, measurements_trend_bai = (
            bai_find_the_best_arm_exact(current_circuit, H_qubit_op, generator_pool, fragment_group_indices_map, commutator_indices_map, iteration, n_qubits))

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
        def vqe_obj(x):
            circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops, x)
            energy = measure_expectation(circuit, H_sparse_pauli_op, shots=1024)
            return energy

        # Minimal debugging for first iteration only
        if verbose and iteration == 0 and len(ansatz_ops) == 1:
            print(f"  Testing objective function at θ=0: E = {vqe_obj([0.0]):.8f}")

        # Try optimization with different methods and starting points
        initial_guess = params.copy()
        print(f"  Starting optimization from: {initial_guess}")

        # Use faster optimization with fewer function evaluations for large systems
        max_iter = 50 if len(ansatz_ops) > 5 else 100
        if len(ansatz_ops) == 1:
            # For single parameter, try simple line search first
            test_params = [-0.1, -0.05, 0.0, 0.05, 0.1]
            best_energy = float('inf')
            best_param = 0.0
            for test_val in test_params:
                energy = vqe_obj([test_val])
                if energy < best_energy:
                    best_energy = energy
                    best_param = test_val
            initial_guess = [best_param]

        # Use faster optimization method
        res = minimize(vqe_obj, initial_guess, method='Powell',
                      options={'maxiter': max_iter, 'disp': False})

        params = list(res.x)

        # Update energy
        final_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops, params)
        energy = measure_expectation(final_circuit, H_sparse_pauli_op)
        energies.append(energy)

        if verbose:
            print(f"  Energy after iteration {iteration}: {energy:.8f}")

    # Return final state
    final_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops, params)
    final_state = get_statevector(final_circuit)

    if verbose:
        print(f"Total measurements used: {total_measurements}")

    return energies, params, ansatz_ops, final_state, total_measurements, total_measurements_at_each_step, total_measurements_trend_bai

if __name__ == "__main__":
    if len(sys.argv) < 6:
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

    energies, params, ansatz, final_state, total_measurements, total_measurements_at_each_step, total_measurements_trend_bai = adapt_vqe_qiskit(
        H_sparse_pauli_op, n_qubits, n_electrons, H_qubit_op, generator_pool,
        fragment_group_indices_map, commutator_indices_map)

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
