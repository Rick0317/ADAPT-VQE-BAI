# Hartree-Fock State Validation Guide

## Overview

This guide explains how to validate that the Hartree-Fock state is correctly constructed in the ADAPT-VQE implementation. The Hartree-Fock state serves as the initial reference state for the variational quantum eigensolver, so it's crucial to ensure it's correct.

## What is the Hartree-Fock State?

The Hartree-Fock state is the ground state of the non-interacting (mean-field) Hamiltonian. In the context of quantum chemistry:

1. **Fermionic representation**: It's a Slater determinant with the lowest-energy orbitals occupied
2. **Qubit representation**: It's a computational basis state that corresponds to the occupation pattern after applying the Bravyi-Kitaev (BK) or Jordan-Wigner (JW) transformation

## How the Hartree-Fock State is Constructed

In the current implementation (`adapt_vqe_exact.py`), the Hartree-Fock state is constructed as follows:

```python
# 1. Get the occupation number pattern
ref_occ = get_occ_no(mol, n_qubits)
# For H4 with 4 electrons: ref_occ = "11110000"

# 2. Convert to qubit basis state using Bravyi-Kitaev transformation
hf_state = get_reference_state(ref_occ, gs_format='wfs')
```

### Step-by-step process:

1. **Occupation pattern**: `get_occ_no(mol, n_qubits)` creates a string like `"11110000"` where:
   - `1` represents occupied orbitals (electrons present)
   - `0` represents unoccupied orbitals (electrons absent)
   - The pattern is ordered from lowest to highest energy orbitals

2. **Bravyi-Kitaev transformation**: `get_reference_state()` applies the BK transformation:
   - Converts the fermionic occupation pattern to a qubit basis state
   - Uses the transformation matrix defined in `get_bk_tf_matrix()`
   - Returns a state vector with a single non-zero element (computational basis state)

3. **State vector**: The result is a normalized state vector in the computational basis

## Validation Checks

The `validate_hartree_fock_state()` function performs comprehensive validation:

### 1. State Normalization
```python
norm = np.linalg.norm(hf_state)
print(f"State normalization: {norm:.10f} (should be 1.0)")
```
- **Expected**: `norm ≈ 1.0`
- **Issue**: If not normalized, the state may have been corrupted

### 2. Occupation Pattern Check
```python
ref_occ = get_occ_no(mol, n_qubits)
print(f"Expected occupation pattern: {ref_occ}")
```
- **Expected**: Pattern matches the molecule's electron count
- **Example**: H4 with 4 electrons should have `"11110000"`

### 3. Basis State Correspondence
```python
bk_basis_state = get_bk_basis_states(ref_occ, n_qubits)
expected_index = find_index(bk_basis_state)
max_amplitude_idx = np.argmax(np.abs(hf_state))
```
- **Expected**: `max_amplitude_idx == expected_index`
- **Issue**: If different, the BK transformation may be incorrect

### 4. Computational Basis State
```python
non_zero_elements = np.sum(np.abs(hf_state) > 1e-10)
```
- **Expected**: `non_zero_elements == 1`
- **Issue**: If > 1, the state is not a computational basis state

### 5. Energy Validation
```python
hf_energy = measure_expectation_statevector(Statevector(hf_state), H_sparse_pauli_op)
```
- **Expected**: `hf_energy > exact_energy` (Hartree-Fock energy should be above exact)
- **Issue**: If `hf_energy < exact_energy`, there may be a problem with the state or Hamiltonian

### 6. Reference Energy Comparison
```python
known_hf_energies = {
    'h2': -1.137,
    'lih': -7.88,
    'beh2': -15.0,
    'h4': -2.0,
}
```
- **Expected**: Energy close to known reference values
- **Tolerance**: Within 0.1 Hartree

## Common Issues and Solutions

### Issue 1: Wrong Basis State
**Symptoms**: `max_amplitude_idx != expected_index`
**Causes**:
- Incorrect Bravyi-Kitaev transformation
- Wrong occupation pattern
- Index calculation error

**Solutions**:
1. Check the BK transformation matrix
2. Verify the occupation pattern for the molecule
3. Debug the `find_index()` function

### Issue 2: Not a Computational Basis State
**Symptoms**: `non_zero_elements > 1`
**Causes**:
- State preparation error
- Normalization issues
- Numerical precision problems

**Solutions**:
1. Check the `get_reference_state()` function
2. Ensure proper normalization
3. Increase numerical precision

### Issue 3: Hartree-Fock Energy Below Exact Energy
**Symptoms**: `hf_energy < exact_energy`
**Causes**:
- Incorrect Hamiltonian
- Wrong state preparation
- Numerical errors

**Solutions**:
1. Verify the Hamiltonian construction
2. Check the state preparation
3. Validate the exact energy calculation

## Testing the Validation

You can test the validation using the provided test script:

```bash
python test_hf_validation.py
```

This script will:
1. Load a test Hamiltonian (if available)
2. Create a Hartree-Fock state
3. Run the validation checks
4. Report any issues

## Integration with ADAPT-VQE

The validation is automatically called in `adapt_vqe_exact.py` after the Hartree-Fock state is created:

```python
# Prepare Hartree-Fock state
ref_occ = get_occ_no(mol, n_qubits)
hf_state = get_reference_state(ref_occ, gs_format='wfs')

# Validate the Hartree-Fock state
validation_results = validate_hartree_fock_state(hf_state, H_sparse_pauli_op, mol, n_qubits, n_electrons, exact_energy)

# Check if validation passed
if not validation_results['correct_basis_state']:
    print("WARNING: Hartree-Fock state validation failed!")
```

## Debugging Tips

1. **Print intermediate values**: Add print statements to see the occupation pattern, BK basis state, and expected index
2. **Check molecule parameters**: Verify that `n_electrons` and `n_qubits` are correct for the molecule
3. **Compare with known results**: Use reference energies from literature
4. **Test with simple cases**: Start with H2 or H4 which have well-known properties

## References

1. Bravyi-Kitaev transformation: arXiv:quant-ph/0003137
2. ADAPT-VQE algorithm: https://doi.org/10.1021/acs.jctc.8b00450
3. Quantum chemistry with quantum computers: https://doi.org/10.1038/s41567-019-0704-4




