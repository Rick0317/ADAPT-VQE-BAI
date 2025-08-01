import numpy as np
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.linalg import expm
import pickle
import os
import sys
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import csv
from datetime import datetime

# --- OpenFermion imports for chemistry mapping ---
from openfermion import FermionOperator, bravyi_kitaev, get_sparse_operator

# Add path to import from parent directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ferm_utils import ferm_to_qubit
from utils.reference_state_utils import get_reference_state, get_occ_no


def save_results_to_csv(final_energy, total_measurements, exact_energy, fidelity,
                       molecule_name, n_qubits, n_electrons, pool_size,
                       use_parallel, executor_type, max_workers,
                       ansatz_depth,  total_measurements_at_each_step=[], total_measurements_trend_bai=[], energy_at_each_step=[], filename='adapt_vqe_results.csv'):
    """
    Save ADAPT-VQE results to a CSV file.

    Args:
        final_energy: Final energy from ADAPT-VQE
        total_measurements: Total number of measurements used
        exact_energy: Exact ground state energy from diagonalization
        fidelity: Fidelity with exact ground state
        molecule_name: Name of the molecule
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        pool_size: Size of the operator pool
        use_parallel: Whether parallel evaluation was used
        executor_type: Type of executor used ('process' or 'thread')
        max_workers: Maximum number of workers used
        ansatz_depth: Final ansatz depth (number of operators)
        filename: CSV filename to save results
    """

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)

    # Prepare the row data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    energy_error = abs(final_energy - exact_energy)

    row_data = {
        'timestamp': timestamp,
        'molecule': molecule_name,
        'n_qubits': n_qubits,
        'n_electrons': n_electrons,
        'pool_size': pool_size,
        'final_energy': final_energy,
        'exact_energy': exact_energy,
        'energy_error': energy_error,
        'fidelity': fidelity,
        'total_measurements': total_measurements,
        'ansatz_depth': ansatz_depth,
        'use_parallel': use_parallel,
        'executor_type': executor_type if use_parallel else 'serial',
        'max_workers': max_workers if use_parallel else 1,
        'energy_at_each_step': energy_at_each_step,
        'total_measurements_at_each_step': total_measurements_at_each_step,
        'total_measurements_trend_bai':total_measurements_trend_bai
    }

    # Write to CSV
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(row_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow(row_data)

    print(f"Results saved to {filename}")

def compute_gradient(H_sparse, op_sparse, state):
    """Compute gradient ⟨ψ|[H,G]|ψ⟩ = ⟨ψ|HG - GH|ψ⟩"""
    HP_psi = H_sparse.dot(op_sparse.dot(state))
    PH_psi = op_sparse.dot(H_sparse.dot(state))
    return np.vdot(state, HP_psi) - np.vdot(state, PH_psi)

def get_hf_bitstring(n_qubits, n_electrons):
    # Returns a bitstring with n_electrons ones at the left
    return '1' * n_electrons + '0' * (n_qubits - n_electrons)


def hf_circuit(n_qubits, n_electrons, mol='h4'):
    # Prepare Hartree-Fock state using reference state utilities
    ref_occ = get_occ_no(mol, n_qubits)
    hf_state = get_reference_state(ref_occ, gs_format='wfs')

    # Create circuit and initialize with the HF state
    qc = QuantumCircuit(n_qubits)
    # Normalize the state vector to avoid precision issues
    hf_state_normalized = hf_state / np.linalg.norm(hf_state)
    qc.initialize(hf_state_normalized, range(n_qubits))
    return qc


def get_statevector(qc):
    backend = Aer.get_backend('statevector_simulator')
    t_qc = transpile(qc, backend)
    result = backend.run(t_qc).result()
    return Statevector(result.get_statevector(t_qc))


def exact_ground_state_energy(H_sparse):
    vals, vecs = eigsh(H_sparse, k=1, which='SA')
    return vals[0], vecs[:, 0]


def openfermion_qubitop_to_sparsepauliop(q_op, n_qubits):
    """
    Convert OpenFermion Qubit Operators to SparsePauliOp
    :param q_op:
    :param n_qubits:
    :return:
    """
    paulis = []
    coeffs = []
    for term, coeff in q_op.terms.items():
        pauli_str = ['I'] * n_qubits
        for idx, p in term:
            pauli_str[idx] = p
        paulis.append(''.join(pauli_str))
        coeffs.append(coeff)
    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


def fermion_operator_to_qiskit_operator(ferm_op, n_qubits):
    """Convert a FermionOperator to a Qiskit Operator via bravyi_kitaev mapping"""
    qubit_op = ferm_to_qubit(ferm_op)
    sparse_matrix = get_sparse_operator(qubit_op, n_qubits)
    return Operator(sparse_matrix.toarray())


def measure_pauli_expectation(circuit, pauli_op, shots=8192):
    """Measure expectation value of a Pauli operator using shot-based simulation"""
    # Create measurement circuit for each Pauli term
    total_expectation = 0.0
    counts = {}
    for i, (pauli_string, coeff) in enumerate(pauli_op.to_list()):
        if pauli_string == 'I' * len(pauli_string):
            # Identity term contributes coefficient directly
            total_expectation += coeff
            continue

        if i == 0:
            # Create measurement circuit for this Pauli string
            meas_circuit = circuit.copy()
            meas_circuit.add_register(ClassicalRegister(len(pauli_string), 'c'))

            # Apply basis rotations for measurement
            # Note: Qiskit Pauli strings use reverse indexing (leftmost = highest qubit)
            n_qubits = len(pauli_string)
            for i, pauli in enumerate(pauli_string):
                qubit_idx = n_qubits - 1 - i  # Reverse the qubit indexing
                if pauli == 'X':
                    meas_circuit.ry(-np.pi/2, qubit_idx)
                elif pauli == 'Y':
                    meas_circuit.rx(np.pi/2, qubit_idx)
                # Z measurement requires no rotation

            # Add measurements
            for i in range(len(pauli_string)):
                meas_circuit.measure(i, i)

            # Run simulation
            simulator = AerSimulator()
            compiled_circuit = transpile(meas_circuit, simulator)
            result = simulator.run(compiled_circuit, shots=shots).result()
            counts = result.get_counts()

        # Calculate expectation value for this Pauli term
        expectation = 0.0
        for bitstring, count in counts.items():
            # Calculate parity (-1)^(number of 1s in positions where Pauli is not I)
            parity = 1
            for i, pauli in enumerate(pauli_string):
                if pauli != 'I' and bitstring[i] == '1':  # bitstring[i] corresponds to pauli_string[i]
                    parity *= -1
            expectation += parity * count / shots

        total_expectation += coeff * expectation

    return np.real(total_expectation)

def measure_expectation(circuit, observable, backend=None, shots=8192):
    """Measure expectation value of an observable given a quantum circuit"""
    if backend is None or isinstance(backend, type(Aer.get_backend('statevector_simulator'))):
        # Use statevector simulator for exact results
        state = get_statevector(circuit)
        return np.real(state.expectation_value(observable))
    else:
        # Use shot-based simulation
        return measure_pauli_expectation(circuit, observable, 8192)

def create_ansatz_circuit(n_qubits, n_electrons, operators, parameters, mol='h4'):
    """Create a parameterized ansatz circuit"""
    circuit = hf_circuit(n_qubits, n_electrons, mol=mol)
    for op, theta in zip(operators, parameters):
        # Apply exp(i * theta * op) to the circuit
        # Compute matrix exponential: exp(i * theta * G)
        unitary_matrix = expm(theta * op.data)
        unitary_gate = Operator(unitary_matrix)
        circuit.append(unitary_gate, range(n_qubits))
    return circuit
