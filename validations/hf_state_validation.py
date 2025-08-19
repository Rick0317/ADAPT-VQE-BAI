from utils.reference_state_utils import get_occ_no, get_bk_basis_states, find_index
import numpy as np
from qiskit.quantum_info import Statevector
from adaptvqe.adapt_vqe_preparation import measure_expectation_statevector


def validate_hartree_fock_state(hf_state, H_sparse_pauli_op, mol, n_qubits,
                                n_electrons, exact_energy=None):
    """
    Comprehensive validation of the Hartree-Fock state.

    Args:
        hf_state: Hartree-Fock state vector
        H_sparse_pauli_op: Hamiltonian as SparsePauliOp
        mol: Molecule name
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        exact_energy: Exact ground state energy (optional)

    Returns:
        dict: Validation results
    """
    print("\n=== Hartree-Fock State Validation ===")

    # 1. Check state normalization
    norm = np.linalg.norm(hf_state)
    print(f"1. State normalization: {norm:.10f} (should be 1.0)")
    if abs(norm - 1.0) > 1e-10:
        print(f"   WARNING: State is not normalized! Normalizing...")
        hf_state = hf_state / norm

    # 2. Check occupation number pattern
    ref_occ = get_occ_no(mol, n_qubits)
    print(f"2. Expected occupation pattern: {ref_occ}")
    print(f"   Number of electrons: {n_electrons}")
    print(f"   Number of qubits: {n_qubits}")

    # 3. Check if state corresponds to the expected basis state
    bk_basis_state = get_bk_basis_states(ref_occ, n_qubits)
    expected_index = find_index(bk_basis_state)
    print(f"3. Expected Bravyi-Kitaev basis state: {bk_basis_state}")
    print(f"   Expected basis index: {expected_index}")

    # Find the actual basis state with maximum amplitude
    max_amplitude_idx = np.argmax(np.abs(hf_state))
    max_amplitude = np.abs(hf_state[max_amplitude_idx])
    print(
        f"   Actual maximum amplitude: {max_amplitude:.10f} at index {max_amplitude_idx}")

    if max_amplitude_idx == expected_index:
        print(f"   ✓ State correctly corresponds to expected basis state")
    else:
        print(
            f"   ✗ WARNING: State does not correspond to expected basis state!")
        print(
            f"     Expected index: {expected_index}, Actual index: {max_amplitude_idx}")

    # 4. Check if state is a computational basis state (should have only one non-zero element)
    non_zero_elements = np.sum(np.abs(hf_state) > 1e-10)
    print(f"4. Number of non-zero elements: {non_zero_elements}")
    if non_zero_elements == 1:
        print(f"   ✓ State is a computational basis state")
    else:
        print(f"   ✗ WARNING: State is not a computational basis state!")
        print(f"     This might indicate an issue with the state preparation")

    # 5. Calculate Hartree-Fock energy
    hf_energy = measure_expectation_statevector(Statevector(hf_state),
                                                H_sparse_pauli_op)
    print(f"5. Hartree-Fock energy: {hf_energy:.8f}")

    # 6. Compare with exact energy if provided
    if exact_energy is not None:
        energy_diff = hf_energy - exact_energy
        print(f"6. Energy difference from exact: {energy_diff:.8f}")
        if energy_diff > 0:
            print(
                f"   ✓ Hartree-Fock energy is above exact energy (as expected)")
        else:
            print(f"   ✗ WARNING: Hartree-Fock energy is below exact energy!")
            print(
                f"     This might indicate an issue with the state or Hamiltonian")

    # 7. Check particle number conservation (if applicable)
    # This would require constructing the particle number operator
    print(
        f"7. Particle number conservation check: Skipped (requires particle number operator)")

    # 8. Check if state is in the correct symmetry sector
    print(f"8. Symmetry sector check: Skipped (requires symmetry operators)")

    # 9. Validate against known reference values for common molecules
    known_hf_energies = {
        'h2': -1.137,
        # Approximate values - should be replaced with actual values
        'lih': -7.88,
        'beh2': -15.0,
        'h4': -2.0,
    }

    if mol.lower() in known_hf_energies:
        expected_hf = known_hf_energies[mol.lower()]
        energy_diff_ref = abs(hf_energy - expected_hf)
        print(f"9. Comparison with reference HF energy for {mol}:")
        print(f"   Expected: {expected_hf:.6f}, Actual: {hf_energy:.6f}")
        print(f"   Difference: {energy_diff_ref:.6f}")
        if energy_diff_ref < 0.1:  # Allow some tolerance
            print(f"   ✓ Energy is close to reference value")
        else:
            print(f"   ⚠ Energy differs significantly from reference value")
    else:
        print(f"9. No reference energy available for {mol}")

    print("=== End of Hartree-Fock State Validation ===\n")

    # Return validation results
    validation_results = {
        'normalized': abs(norm - 1.0) < 1e-10,
        'correct_basis_state': max_amplitude_idx == expected_index,
        'computational_basis_state': non_zero_elements == 1,
        'hf_energy': hf_energy,
        'energy_above_exact': exact_energy is None or hf_energy > exact_energy,
        'max_amplitude': max_amplitude,
        'expected_index': expected_index,
        'actual_index': max_amplitude_idx
    }

    return validation_results
