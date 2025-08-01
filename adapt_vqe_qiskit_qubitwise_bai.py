import numpy as np
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
from scipy.optimize import minimize
import pickle
import os
import sys
import numpy as np
from multiprocessing import Pool, cpu_count, get_context
from functools import partial
import csv
from datetime import datetime
import json

# --- OpenFermion imports for chemistry mapping ---

# Add path to import from parent directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ferm_utils import ferm_to_qubit
from utils.qubit_utils import get_commutator_qubit, qubit_operator_to_qiskit_operator
from utils.decomposition import qwc_decomposition
from utils.reference_state_utils import get_reference_state, get_occ_no
from adaptvqe.adapt_vqe_preparation import (create_ansatz_circuit, measure_expectation,
                              get_statevector, openfermion_qubitop_to_sparsepauliop,
                              exact_ground_state_energy, save_results_to_csv)
import gc, psutil
from get_generator_pool import get_generator_pool
import multiprocessing

# Set multiprocessing start method to 'spawn' for better memory management
multiprocessing.set_start_method("spawn", force=True)


def pauli_commute(p1, p2):
    # Returns True if p1 and p2 commute on a single qubit
    if p1 == 'I' or p2 == 'I' or p1 == p2:
        return True
    # Check anti-commuting pairs
    anti_commuting_pairs = {('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y')}
    return (p1, p2) not in anti_commuting_pairs


def does_commute(pauli_string1, pauli_string2):
    if len(pauli_string1) != len(pauli_string2):
        raise ValueError("Pauli strings must have the same length")
    return all(pauli_commute(p1, p2) for p1, p2 in zip(pauli_string1, pauli_string2))


def does_diagonalize(group_op, key_unitary):
    """
    We check if key_unitary diagonalizes all the Pauli Strings in group_op
    :param group_op: The QWC group operator
    :param key_unitary: The diagonalizing unitary in the form of 'IXYI' where X and Y are rotations to diagonalize X and Y respectively
    :return:
    """
    for pauli_string, coeff in group_op.to_list():
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X' and key_unitary[i] != 'X':
                return False
            elif pauli == 'Y' and key_unitary[i] != 'Y':
                return False
            elif pauli == 'Z' and key_unitary[i] != 'I':
                return False
    return True


def get_diagonalizing_pauli(group_op, n_qubits):
    """
    Given a QWC group operator, find the diagonalizing Pauli String that diagonalizes all the Pauli Strings in group_op
    :param group_op: SparsePauliOp representing a QWC group
    :return: Pauli string indicating diagonalizing rotations ('X' for Ry, 'Y' for Rx, 'I' for no rotation)
    """
    # Get all Pauli strings from the group
    pauli_list = group_op.to_list()
    if not pauli_list:
        return 'I'  # Empty group

    # Initialize diagonalizing string with all identities
    diagonalizing_pauli = ['I'] * n_qubits

    # For each qubit position, determine what rotation is needed
    for pauli_string, coeff in pauli_list:
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                # X basis requires Ry(-π/2) rotation, represented as 'X'
                diagonalizing_pauli[i] = 'X'
            elif pauli == 'Y':
                # Y basis requires Rx(π/2) rotation, represented as 'Y'
                diagonalizing_pauli[i] = 'Y'
            # 'Z' and 'I' don't require rotations, so keep as 'I'

    return ''.join(diagonalizing_pauli)

def _process_single_operator(args):
    """Worker function for parallel commutator processing"""
    i, H_qubit_op, qubit_generator, n_qubits = args

    try:
        # Clear memory before starting
        gc.collect()

        # Convert to qubit operator
        commutator_qubit = get_commutator_qubit(H_qubit_op, qubit_generator)

        # Decompose into Pauli groups
        pauli_groups = qwc_decomposition(commutator_qubit)

        # Clean up
        del commutator_qubit
        gc.collect()

        # Store both the group data and pauli strings for does_diagonalize checking
        groups_data = []
        for group in pauli_groups:
            group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)
            # Store the pauli strings and coefficients instead of the SparsePauliOp object
            group_pauli_list = group_op.to_list()
            groups_data.append(group_pauli_list)
            # Clean up each group after processing
            del group_op

        # Final cleanup
        del pauli_groups
        gc.collect()

        return i, groups_data

    except Exception as e:
        print(f"Error processing operator {i}: {e}")
        gc.collect()  # Clean up on error too
        return i, []


def get_qubit_w_commutation_group_parallel(H_qubit_op, generator_pool, n_qubits, max_processes=None):
    """
    Parallelized version of get_qubit_w_commutation_group
    Categorizes each QWC group and express each fragment with the categories
    :param H_fermion: Hamiltonian in Fermion Operator
    :param pool_fermion: The pool of fermion generators
    :param n_qubits: The number of qubits
    :param max_processes: Maximum number of processes to use (None for auto)
    :return: fragment_group_indices_map, commutator_indices_map
    """

    # Step 1: Parallel computation of all commutators and QWC groups
    print(f"Computing commutators for {len(generator_pool)} operators in parallel...")

    args_list = [(i, H_qubit_op, qubit_generator, n_qubits)
                 for i, qubit_generator in enumerate(generator_pool)]

    # Determine number of processes (limit for memory management)
    available_processes = cpu_count()
    if max_processes is not None:
        num_processes = min(max_processes, available_processes, len(args_list))
    else:
        # Use fewer processes for large pools to manage memory
        if len(generator_pool) > 200:
            num_processes = min(4, available_processes)  # Limit to 4 for large pools
        else:
            num_processes = min(available_processes, len(args_list))

    print(f"Using {num_processes} processes for commutator computation (out of {available_processes} available)...")

    # Use multiprocessing to compute all commutators in parallel
    try:
        with Pool(num_processes) as p:
            operator_groups = p.map(_process_single_operator, args_list)
        print("Parallel commutator computation completed successfully")
    except Exception as e:
        print(f"Parallel processing failed ({e}), falling back to sequential processing")
        # Fallback to sequential processing
        operator_groups = []
        for args in args_list:
            result = _process_single_operator(args)
            operator_groups.append(result)

    # Step 2: Optimized sequential grouping with batching and fast pre-filtering
    print("Organizing QWC groups using optimized does_diagonalize...")
    fragment_group_indices_map = {}
    commutator_indices_map = {}

    # Process operators in smaller batches to manage memory
    batch_size = 25  # Process 25 operators at a time
    total_operators = len(operator_groups)

    for batch_start in range(0, total_operators, batch_size):
        batch_end = min(batch_start + batch_size, total_operators)
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_operators-1)//batch_size + 1} (operators {batch_start}-{batch_end-1})")

        # Force cleanup before each batch
        gc.collect()

        # Process this batch
        for idx in range(batch_start, batch_end):
            i, groups_data = operator_groups[idx]
            commutator_indices_map[i] = []

            # Process each QWC group for this operator
            for group_pauli_list in groups_data:
                # Fast pre-filtering: get diagonalizing pauli first (cheaper)
                group_op = SparsePauliOp.from_list(group_pauli_list)
                diagonalizing_pauli = get_diagonalizing_pauli(group_op, n_qubits)

                # Fast check: exact match first (most common case)
                if diagonalizing_pauli in fragment_group_indices_map:
                    group_index = fragment_group_indices_map[diagonalizing_pauli]
                    commutator_indices_map[i].append(group_index)
                    del group_op
                    continue

                # Slower check: does_diagonalize for remaining cases
                found_commuting_group = False
                # Limit the search to prevent memory explosion
                max_groups_to_check = len(fragment_group_indices_map)
                groups_checked = 0

                for key_unitary, group_index in fragment_group_indices_map.items():
                    if groups_checked >= max_groups_to_check:
                        break
                    groups_checked += 1

                    # Skip exact matches (already checked above)
                    if key_unitary == diagonalizing_pauli:
                        continue

                    if does_diagonalize(group_op, key_unitary):
                        commutator_indices_map[i].append(group_index)
                        found_commuting_group = True
                        break

                # If no existing unitary works, create a new group
                if not found_commuting_group:
                    new_group_index = len(fragment_group_indices_map)
                    fragment_group_indices_map[diagonalizing_pauli] = new_group_index
                    commutator_indices_map[i].append(new_group_index)

                # Immediate cleanup
                del group_op

            # Cleanup after each operator
            if i % 10 == 0:
                gc.collect()

        # Force cleanup after each batch
        print(f"  Batch completed. Current QWC groups: {len(fragment_group_indices_map)}")
        gc.collect()

    print(f"Found {len(fragment_group_indices_map)} unique QWC groups")
    return fragment_group_indices_map, commutator_indices_map


def save_commutator_maps(fragment_group_indices_map, commutator_indices_map, filename):
    """Save commutator maps to a JSON file"""
    try:
        # Convert all keys to strings for JSON serialization
        data = {
            'fragment_group_indices_map': {str(k): v for k, v in fragment_group_indices_map.items()},
            'commutator_indices_map': {str(k): v for k, v in commutator_indices_map.items()}
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Commutator maps saved to {filename}")

    except Exception as e:
        print(f"Error saving commutator maps: {e}")


def load_commutator_maps(filename):
    """Load commutator maps from a JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        # Convert string keys back to appropriate types
        fragment_group_indices_map = {k: v for k, v in data['fragment_group_indices_map'].items()}
        commutator_indices_map = {int(k): v for k, v in data['commutator_indices_map'].items()}

        print(f"Commutator maps loaded from {filename}")
        return fragment_group_indices_map, commutator_indices_map

    except FileNotFoundError:
        print(f"Commutator maps file {filename} not found")
        return None, None
    except Exception as e:
        print(f"Error loading commutator maps: {e}")
        return None, None


def generate_commutator_cache_filename(molecule_name, n_qubits, n_electrons, pool_size, pool_type):
    """Generate a cache filename based on molecular system parameters"""
    return f"commutator_maps_{molecule_name}_{n_qubits}q_{n_electrons}e_{pool_size}_{pool_type}.json"


def get_commutator_maps_cached(H_qubit_op, generator_pool, n_qubits, molecule_name='h4', n_electrons=4, cache_file=None):
    """
    Get commutator maps with caching support

    Args:
        H_fermion: Hamiltonian fermion operator
        pool_fermion: List of fermion operators in the pool
        n_qubits: Number of qubits
        molecule_name: Name of molecule (for caching)
        n_electrons: Number of electrons (for caching)
        cache_file: Optional filename to cache/load commutator maps

    Returns:
        tuple: (fragment_group_indices_map, commutator_indices_map)
    """
    if cache_file is not None:
        # Try to load from cache first
        fragment_map, commutator_map = load_commutator_maps(cache_file)
        if fragment_map is not None and commutator_map is not None:
            # Verify the cache matches current pool size
            if len(commutator_map) == len(generator_pool):
                return fragment_map, commutator_map
            else:
                print(f"Cache size mismatch: expected {len(generator_pool)}, got {len(commutator_map)}")

    # Compute commutator maps if not cached or cache invalid
    print("Computing commutator maps...")
    # Use fewer processes for memory management with large pools
    max_processes = cpu_count() - 4
    fragment_map, commutator_map = get_qubit_w_commutation_group_parallel(H_qubit_op, generator_pool, n_qubits, max_processes)

    # Save to cache if filename provided
    if cache_file is not None:
        save_commutator_maps(fragment_map, commutator_map, cache_file)

    return fragment_map, commutator_map

def build_measurement_circuit_with_args(args):
    current_circuit, pauli_string, fragment_index, n_qubits = args
    print(f"Building measurement circuit for {fragment_index}")

    meas_circuit = current_circuit.copy()
    creg = ClassicalRegister(len(pauli_string))
    meas_circuit.add_register(creg)

    for i, pauli in enumerate(pauli_string):
        q = n_qubits - 1 - i
        if pauli == 'I':
            continue
        elif pauli == 'X':
            meas_circuit.ry(-np.pi / 2, q)
        elif pauli == 'Y':
            meas_circuit.rx(np.pi / 2, q)

        meas_circuit.measure(q, i)

    return (fragment_index, meas_circuit)


def build_all_circuits_parallel(current_circuit, fragment_group_indices_map, active_qwc_groups, n_qubits):
    # Prepare argument list
    args_list = [
        (current_circuit, pauli_string, fragment_index, n_qubits)
        for pauli_string, fragment_index in fragment_group_indices_map.items()
        if fragment_index in active_qwc_groups
    ]

    with get_context("spawn").Pool(processes=min(cpu_count() - 2, len(args_list))) as pool:
        results = pool.map(build_measurement_circuit_with_args, args_list)

    # Separate outputs
    fragment_indices = [frag_idx for frag_idx, _ in results]
    circuits = [circuit for _, circuit in results]

    return fragment_indices, circuits


def _measure_single_fragment(args):
    """Worker function for parallel measurement of a single fragment"""
    current_circuit, pauli_string, fragment_index, n_qubits, shots, simulator, = args

    try:
        print(f"Measuring fragment {fragment_index}")
        # Create measurement circuit
        meas_circuit = QuantumCircuit(n_qubits, n_qubits)
        meas_circuit.compose(current_circuit, inplace=True)

        # Apply basis rotations for measurement
        # Note: Qiskit Pauli strings use reverse indexing (leftmost = highest qubit)
        for i, pauli in enumerate(pauli_string):
            qubit_idx = n_qubits - 1 - i  # Reverse the qubit indexing
            if pauli == 'X':
                meas_circuit.ry(-np.pi / 2, qubit_idx)
            elif pauli == 'Y':
                meas_circuit.rx(np.pi / 2, qubit_idx)
            # Z measurement requires no rotation

        meas_circuit.measure_all(inplace=True)

        # Run simulation
        compiled = transpile(meas_circuit, simulator, optimization_level=1)

        result = simulator.run(compiled, shots=shots).result()
        counts = result.get_counts()
        del meas_circuit, compiled, result
        return fragment_index, counts

    except Exception as e:
        print(f"Error measuring fragment {fragment_index} with Pauli string {pauli_string}: {e}")
        return fragment_index, {}


def get_counts_for_each_fragment(current_circuit, fragment_group_indices_map, active_qwc_groups, n_qubits, shots=8192):
    """
    Batched construction + simulation to keep memory usage low.
    """
    simulator = AerSimulator()

    # Prepare full args list
    args_list = [
        (current_circuit, pauli_string, fragment_index, n_qubits)
        for pauli_string, fragment_index in fragment_group_indices_map.items()
        if fragment_index in active_qwc_groups
    ]

    counts_for_each_fragment = {}

    # Build circuits in parallel
    with get_context("spawn").Pool(processes=cpu_count() - 2) as pool:
        results = pool.map(build_measurement_circuit_with_args, args_list)

    # Separate fragment indices and circuits
    fragment_indices = [frag_idx for frag_idx, _ in results]
    circuits = [circuit for _, circuit in results]

    # Transpile and simulate this batch
    transpiled_circuits = transpile(circuits, simulator, optimization_level=1)
    result = simulator.run(transpiled_circuits, shots=shots).result()

    for i, fragment_index in enumerate(fragment_indices):
        counts_for_each_fragment[fragment_index] = result.get_counts(i)

    # Clean up this batch
    del circuits, transpiled_circuits, result, results
    gc.collect()

    return counts_for_each_fragment

def compute_commutator_gradient(H_qubit_op, generator_op, fragment_indices, counts_for_each_fragment, n_qubits, shots=8192):
    """
    Compute the energy gradient of the generator
    :param H_fermion: Fermion Operator of the Hamiltonian
    :param generator_ferm: Fermion Operator of the generator
    :param fragment_indices: The QWC fragments indices of the decomposed [H, G]
    :param counts_for_each_fragment: The measurement information of the state for each fragment
    :return:
    """

    # Convert to qubit operator
    commutator_qubit = get_commutator_qubit(H_qubit_op, generator_op)

    # Decompose into QWC groups
    pauli_groups = qwc_decomposition(commutator_qubit)
    total_expectation = 0.0
    for i, group in enumerate(pauli_groups):
        group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)
        fragment_group_index = fragment_indices[i]
        counts = counts_for_each_fragment[fragment_group_index]
        sub_total_expectation = 0.0
        for i, (pauli_string, coeff) in enumerate(group_op.to_list()):
            expectation = 0.0
            for bitstring, count in counts.items():
                # Calculate parity
                parity = 1
                for i, pauli in enumerate(pauli_string):
                    if pauli != 'I' and bitstring[i] == '1':
                        parity *= -1
                expectation += parity * count / shots

            sub_total_expectation += coeff * expectation

        total_expectation += sub_total_expectation

    return total_expectation


def _compute_single_gradient_bai(args):
    """Worker function for parallel gradient computation in BAI"""
    try:
        arm_index, H_qubit_op, generator_op, fragment_indices, counts_for_each_fragment, n_qubits, shots = args

        # Compute gradient for this arm
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

def bai_find_the_best_arm(current_circuit, H_qubit_op, generator_pool, fragment_group_indices_map, commutator_indices_map, iteration, delta= 0.05, max_rounds=10):

    K = len(generator_pool)
    active_arms = list(range(K))
    estimates = np.zeros(K)
    pulls = np.zeros(K)
    active_qwc_groups = set(fragment_group_indices_map.values())
    rounds = 0
    total_shots = 0
    total_measurements_across_fragments = 0
    measurements_trend_bai = []

    print(f"Number of active QWC groups: {len(active_qwc_groups)}")

    while len(active_arms) > 1 and rounds < max_rounds:
        # More aggressive garbage collection for large systems
        gc.collect()
        rounds += 1
        print(f"Round {rounds}")
        shots = 512
        total_shots += shots
        total_measurements_across_fragments += shots * len(active_qwc_groups)
        measurements_trend_bai.append(total_measurements_across_fragments)


        counts_for_each_fragment = get_counts_for_each_fragment(
            current_circuit,
            fragment_group_indices_map,
            active_qwc_groups,
            n_qubits,
            shots=shots,
        )

        # Parallel gradient computation for all active arms
        if len(active_arms) > 1:
            # Prepare arguments for parallel processing
            args_list = [
                (i, H_qubit_op, generator_pool[i], commutator_indices_map[i],
                 counts_for_each_fragment, n_qubits, shots)
                for i in active_arms
            ]

            # Use multiprocessing for parallel gradient computation - scale with system size
            # Limit processes more aggressively for very large systems to avoid memory issues

            num_processes = min(cpu_count() - 4, len(active_arms))

            try:
                print(f"  Starting parallel gradient computation for {len(active_arms)} arms using {num_processes} processes...")
                with Pool(num_processes) as p:
                    gradient_results = p.map(_compute_single_gradient_bai, args_list)

                # Update estimates and pulls based on parallel results
                for arm_index, reward in gradient_results:
                    estimates[arm_index] = (estimates[arm_index] * (total_shots - shots) + reward * shots) / total_shots
                    pulls[arm_index] += shots

                print(f"  Completed gradient computation for {len(active_arms)} arms")

            except Exception as e:
                print(f"Parallel gradient computation failed ({e}), falling back to sequential")
                # Fallback to sequential processing
                for i in active_arms:
                    generator_op = generator_pool[i]
                    rewards = compute_commutator_gradient(H_qubit_op, generator_op, commutator_indices_map[i],
                                                        counts_for_each_fragment, n_qubits, shots)
                    estimates[i] = (estimates[i] * (total_shots - shots) + rewards * shots) / total_shots
                    pulls[i] += shots
        else:
            # Single arm - no need for multiprocessing overhead
            i = active_arms[0]
            generator_op = generator_pool[i]
            rewards = compute_commutator_gradient(H_qubit_op, generator_op, commutator_indices_map[i],
                                                counts_for_each_fragment, n_qubits, shots)
            estimates[i] = (estimates[i] * (total_shots - shots) + rewards * shots) / total_shots
            pulls[i] += shots


        means = estimates
        max_mean = max(abs(means[active_arms]))

        radius = np.sqrt(
            np.log(0.0001 * len(active_arms) * (pulls ** 2) / (delta * 100 * (iteration + 1))) / pulls)

        new_active_arms = []
        new_active_qwc_groups = set()
        for i in active_arms:
            if abs(means[i]) + radius[i] >= max_mean - radius[i]:
                new_active_arms.append(i)
                # The list of QWC fragments that will still be measured
                fragments_list = commutator_indices_map[i]

                # Update the active_qwc_groups set
                new_active_qwc_groups.update(fragments_list)

        active_arms = new_active_arms
        active_qwc_groups = new_active_qwc_groups

        print(f"After round {rounds}, active_arms: {active_arms}")
        print(f"After round {rounds}, number of fragments: {len(active_qwc_groups)}")
        gc.collect()

    means = estimates
    print(f"Final BAI result: {len(active_arms)} active arms remaining after {rounds} rounds")
    best_arm = max(np.array(active_arms), key=lambda i: abs(means[i]))
    print(f"Selected best arm {best_arm} with gradient magnitude {abs(means[best_arm]):.6e}")
    return abs(means[best_arm]), best_arm, total_measurements_across_fragments, measurements_trend_bai


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
            bai_find_the_best_arm(current_circuit, H_qubit_op, generator_pool, fragment_group_indices_map, commutator_indices_map, iteration))

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
