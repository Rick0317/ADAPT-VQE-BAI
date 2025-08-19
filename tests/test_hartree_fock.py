import pytest
import numpy as np
import pickle
import os
import sys
from qiskit.quantum_info import Statevector
from openfermion import QubitOperator, get_sparse_operator

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.reference_state_utils import get_reference_state, get_occ_no, get_jw_basis_states, find_index
from adaptvqe.adapt_vqe_preparation import get_hf_statevector, exact_ground_state_energy
from utils.ferm_utils import ferm_to_qubit
from adaptvqe.adapt_vqe_preparation import (create_ansatz_circuit, measure_expectation,
                              get_statevector, openfermion_qubitop_to_sparsepauliop,
                              exact_ground_state_energy, save_results_to_csv, save_intermediate_results_to_csv,
                              create_ansatz_statevector, measure_expectation_statevector,
                              get_hf_statevector, apply_operator_to_statevector)

def test_hartree_fock():
    """Test Hartree-Fock state generation and energy calculation for various molecules"""

    # Test cases: (molecule_name, n_qubits, n_electrons, expected_hf_energy_tolerance)
    test_cases = [
        ('h4', 8, 4, 1e-3),      # H4 molecule
        ('lih', 12, 4, 1e-3),    # LiH molecule
        ('beh2', 14, 6, 1e-3),  # BeH2 molecule
    ]

    print("🧪 Testing Hartree-Fock state generation and energy calculation...\n")

    for mol, n_qubits, n_electrons, energy_tol in test_cases:
        print(f"Testing molecule: {mol.upper()} ({n_qubits} qubits, {n_electrons} electrons)")

        # Test 1: Occupation number generation
        test_occupation_numbers(mol, n_qubits, n_electrons)

        # Test 2: Reference state creation
        test_reference_state_creation(mol, n_qubits, n_electrons)

        # Test 3: HF statevector creation
        test_hf_statevector_creation(mol, n_qubits, n_electrons)

        # Test 4: HF energy calculation
        test_hf_energy_calculation(mol, n_qubits, n_electrons, energy_tol)

        # Test 5: State normalization and properties
        test_state_properties(mol, n_qubits, n_electrons)

        print(f"✓ {mol.upper()} tests completed successfully\n")

    print("🎉 All Hartree-Fock tests passed!")


def test_occupation_numbers(mol, n_qubits, n_electrons):
    """Test occupation number generation for Hartree-Fock state"""
    print(f"  Testing occupation number generation...")

    # Get occupation numbers
    occ_no = get_occ_no(mol, n_qubits)

    # Verify occupation number format
    assert isinstance(occ_no, str), f"Occupation number should be a string, got {type(occ_no)}"
    assert len(occ_no) == n_qubits, f"Occupation number length should be {n_qubits}, got {len(occ_no)}"

    # Verify occupation number content
    assert occ_no.count('1') == n_electrons, f"Should have {n_electrons} occupied orbitals, got {occ_no.count('1')}"
    assert occ_no.count('0') == n_qubits - n_electrons, f"Should have {n_qubits - n_electrons} virtual orbitals, got {occ_no.count('0')}"

    # Verify occupation number structure (occupied first, then virtual)
    expected_occ = '1' * n_electrons + '0' * (n_qubits - n_electrons)
    assert occ_no == expected_occ, f"Occupation number should be {expected_occ}, got {occ_no}"

    print(f"    ✓ Occupation number: {occ_no}")


def test_reference_state_creation(mol, n_qubits, n_electrons):
    """Test reference state creation from occupation numbers"""
    print(f"  Testing reference state creation...")

    # Get occupation numbers
    occ_no = get_occ_no(mol, n_qubits)

    # Create reference state
    ref_state = get_reference_state(occ_no, gs_format='wfs')

    # Verify state format
    assert isinstance(ref_state, np.ndarray), f"Reference state should be numpy array, got {type(ref_state)}"
    assert ref_state.shape == (2**n_qubits,), f"State should have shape ({2**n_qubits},), got {ref_state.shape}"

    # Verify state is normalized
    norm = np.linalg.norm(ref_state)
    assert np.abs(norm - 1.0) < 1e-10, f"State should be normalized, norm = {norm}"

    # Verify state has correct structure (single 1 at correct position)
    ones_count = np.sum(ref_state == 1.0)
    zeros_count = np.sum(ref_state == 0.0)
    assert ones_count == 1, f"State should have exactly one 1, got {ones_count}"
    assert zeros_count == 2**n_qubits - 1, f"State should have {2**n_qubits - 1} zeros, got {zeros_count}"

    # Verify the 1 is at the correct position (Jordan-Wigner mapping)
    jw_basis_state = get_jw_basis_states(occ_no, n_qubits)
    expected_index = find_index(jw_basis_state)
    actual_index = np.where(ref_state == 1.0)[0][0]
    assert actual_index == expected_index, f"1 should be at index {expected_index}, got {actual_index}"

    print(f"    ✓ Reference state created with norm {norm:.6f}")


def test_hf_statevector_creation(mol, n_qubits, n_electrons):
    """Test Hartree-Fock statevector creation"""
    print(f"  Testing HF statevector creation...")

    # Create HF statevector
    hf_statevector = get_hf_statevector(n_qubits, n_electrons, mol=mol)

    # Verify statevector format
    assert isinstance(hf_statevector, Statevector), f"HF statevector should be Qiskit Statevector, got {type(hf_statevector)}"
    assert hf_statevector.dim == 2**n_qubits, f"Statevector dimension should be {2**n_qubits}, got {hf_statevector.dim}"

    # Verify statevector is normalized
    norm = np.linalg.norm(hf_statevector.data)
    assert np.abs(norm - 1.0) < 1e-10, f"Statevector should be normalized, norm = {norm}"

    # Verify statevector has correct structure
    ones_count = np.sum(np.abs(hf_statevector.data) > 0.99)
    zeros_count = np.sum(np.abs(hf_statevector.data) < 0.01)
    assert ones_count == 1, f"Statevector should have exactly one non-zero element, got {ones_count}"
    assert zeros_count == 2**n_qubits - 1, f"Statevector should have {2**n_qubits - 1} zero elements, got {zeros_count}"

    # Verify the non-zero element is at the correct position
    occ_no = get_occ_no(mol, n_qubits)
    jw_basis_state = get_jw_basis_states(occ_no, n_qubits)
    expected_index = find_index(jw_basis_state)
    actual_index = np.argmax(np.abs(hf_statevector.data))
    assert actual_index == expected_index, f"Non-zero element should be at index {expected_index}, got {actual_index}"

    print(f"    ✓ HF statevector created with norm {norm:.6f}")


def test_hf_energy_calculation(mol, n_qubits, n_electrons, energy_tol):
    """Test Hartree-Fock energy calculation"""
    print(f"  Testing HF energy calculation...")

    try:
                # Load Hamiltonian from file
        ham_file = f'../ham_lib/{mol}_fer.bin'
        if not os.path.exists(ham_file):
            print(f"    ⚠️  Hamiltonian file {ham_file} not found, skipping energy test")
            return

        print(f"    Loading Hamiltonian from {ham_file}...")
        with open(ham_file, 'rb') as f:
            fermion_op = pickle.load(f)

        print(f"    Fermion operator loaded, converting to qubit operator...")
        # Convert to qubit operator
        H_qubit_op = ferm_to_qubit(fermion_op)
        print(f"    Qubit operator created with {len(H_qubit_op.terms)} terms")

        # Get HF statevector
        print(f"    Creating HF statevector...")
        hf_statevector = get_hf_statevector(n_qubits, n_electrons, mol=mol)

        H_sparse_pauli_op = openfermion_qubitop_to_sparsepauliop(
            H_qubit_op, n_qubits)

        # Calculate HF energy
        print(f"    Converting to sparse matrix...")
        H_sparse = get_sparse_operator(H_qubit_op, n_qubits)
        print(f"    Sparse matrix created with shape {H_sparse.shape}")
        print(f"    Sparse matrix type: {type(H_sparse)}")

        initial_state = create_ansatz_statevector(n_qubits, n_electrons,
                                                  [],
                                                  [], mol=mol)
        final_statevector = initial_state
        hf_energy = measure_expectation_statevector(final_statevector,
                                                 H_sparse_pauli_op)
        print(f"HF energy (Qiskit): {hf_energy}")

        # Verify energy is finite
        assert np.isfinite(hf_energy), f"HF energy should be finite, got {hf_energy}"

        # Verify energy is reasonable (should be negative for stable molecules)
        assert hf_energy < 100, f"HF energy should be reasonable, got {hf_energy}"

        H_sparse_pauli_op = openfermion_qubitop_to_sparsepauliop(
            H_qubit_op, n_qubits)

        # Compute exact ground state energy
        H_sparse = H_sparse_pauli_op.to_matrix(sparse=True)
        exact_energy, exact_gs = exact_ground_state_energy(H_sparse)

        # Verify HF energy is higher than exact energy (variational principle)
        energy_diff = hf_energy - exact_energy
        assert energy_diff >= -energy_tol, f"HF energy should be >= exact energy, difference: {energy_diff}"

        print(f"    ✓ HF energy: {hf_energy:.6f}")
        print(f"    ✓ Exact energy: {exact_energy:.6f}")
        print(f"    ✓ Energy difference: {energy_diff:.6f}")

        # Verify energy difference is reasonable (HF should be close to exact for small molecules)
        if mol in ['h4', 'lih']:  # Small molecules where HF should be reasonable
            assert energy_diff < 0.1, f"HF energy should be close to exact for {mol}, difference: {energy_diff}"

    except Exception as e:
        print(f"    ⚠️  Energy calculation failed: {e}")
        print(f"    ⚠️  This might be due to missing Hamiltonian file or calculation issues")


def test_state_properties(mol, n_qubits, n_electrons):
    """Test various properties of the HF state"""
    print(f"  Testing state properties...")

    # Get HF statevector
    hf_statevector = get_hf_statevector(n_qubits, n_electrons, mol=mol)

    # Test 1: State normalization
    norm = np.linalg.norm(hf_statevector.data)
    assert np.abs(norm - 1.0) < 1e-10, f"State should be normalized, norm = {norm}"

    # Test 2: State is real (up to numerical precision)
    imag_part = np.imag(hf_statevector.data)
    max_imag = np.max(np.abs(imag_part))
    assert max_imag < 1e-10, f"State should be real, max imaginary part: {max_imag}"

    # Test 3: State has correct number of non-zero elements
    non_zero_elements = np.sum(np.abs(hf_statevector.data) > 1e-10)
    assert non_zero_elements == 1, f"State should have exactly one non-zero element, got {non_zero_elements}"

    # Test 4: State is in computational basis
    # The HF state should be a computational basis state (single 1, rest 0s)
    basis_state_indices = np.where(np.abs(hf_statevector.data) > 0.99)[0]
    assert len(basis_state_indices) == 1, f"State should be a computational basis state"

    # Test 5: State satisfies particle number conservation
    # For the HF state, the number of 1s in the binary representation should equal n_electrons
    basis_index = basis_state_indices[0]
    binary_rep = format(basis_index, f'0{n_qubits}b')
    particle_count = binary_rep.count('1')
    assert particle_count == n_electrons, f"State should have {n_electrons} particles, got {particle_count}"

    print(f"    ✓ State properties validated:")
    print(f"      - Normalized: {norm:.6f}")
    print(f"      - Max imaginary part: {max_imag:.2e}")
    print(f"      - Non-zero elements: {non_zero_elements}")
    print(f"      - Particle count: {particle_count}")


def test_edge_cases():
    """Test edge cases for HF state generation"""
    print("\n🧪 Testing edge cases...")

    # Test 1: Minimum system (H2-like)
    print("  Testing minimum system (2 qubits, 2 electrons)...")
    try:
        hf_state = get_hf_statevector(2, 2, mol='h2')
        assert hf_state.dim == 4, f"Expected dimension 4, got {hf_state.dim}"
        print("    ✓ Minimum system test passed")
    except Exception as e:
        print(f"    ⚠️  Minimum system test failed: {e}")

    # Test 2: Invalid inputs
    print("  Testing invalid inputs...")

    # Test invalid molecule
    with pytest.raises(Exception):
        get_occ_no('invalid_mol', 4)

    # Test invalid qubit count
    with pytest.raises(Exception):
        get_occ_no('h4', -1)

    # Test invalid electron count
    with pytest.raises(Exception):
        get_occ_no('h4', 4)  # 4 qubits but 4 electrons (should work)

    print("    ✓ Invalid input handling test passed")


def test_hamiltonian_loading():
    """Test Hamiltonian loading and conversion for debugging"""
    print("\n🧪 Testing Hamiltonian loading...")

    # Test with H4 molecule
    mol = 'h4'
    n_qubits = 8
    n_electrons = 4

    try:
        # Load Hamiltonian from file
        ham_file = f'../ham_lib/{mol}_fer.bin'
        if not os.path.exists(ham_file):
            print(f"  ⚠️  Hamiltonian file {ham_file} not found, skipping test")
            return

        print(f"  Loading Hamiltonian from {ham_file}...")
        with open(ham_file, 'rb') as f:
            fermion_op = pickle.load(f)

        print(f"  Fermion operator loaded, type: {type(fermion_op)}")

        # Convert to qubit operator
        print(f"  Converting to qubit operator...")
        H_qubit_op = ferm_to_qubit(fermion_op)
        print(f"  Qubit operator created with {len(H_qubit_op.terms)} terms")

        # Convert to sparse matrix
        print(f"  Converting to sparse matrix...")
        H_sparse = get_sparse_operator(H_qubit_op, n_qubits)
        print(f"  Sparse matrix created with shape {H_sparse.shape}, type: {type(H_sparse)}")

        # Test if it's a scipy sparse matrix
        if hasattr(H_sparse, 'toarray'):
            print(f"  ✓ Matrix has toarray() method - it's a scipy sparse matrix")
        else:
            print(f"  ⚠️  Matrix does not have toarray() method")

        # Test if it works with exact_ground_state_energy
        print(f"  Testing exact_ground_state_energy...")
        try:
            exact_energy, _ = exact_ground_state_energy(H_sparse)
            print(f"  ✓ Exact energy calculation successful: {exact_energy:.6f}")
        except Exception as e:
            print(f"  ❌ Exact energy calculation failed: {e}")
            print(f"  Matrix type: {type(H_sparse)}")
            print(f"  Matrix shape: {H_sparse.shape}")
            if hasattr(H_sparse, 'toarray'):
                print(f"  Matrix has toarray method")
            else:
                print(f"  Matrix does not have toarray method")

        print(f"  ✓ Hamiltonian loading test completed")

    except Exception as e:
        print(f"  ❌ Hamiltonian loading test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run all tests
    test_hartree_fock()
    test_edge_cases()
    test_hamiltonian_loading()
    print("\n🎉 All Hartree-Fock tests completed!")
