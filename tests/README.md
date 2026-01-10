# Operator Pool Tests

This directory contains comprehensive tests for the `get_generator_pool` function, which generates different types of operator pools for ADAPT-VQE.

## Test Coverage

The tests cover two main areas:

### 1. Operator Pool Generation
- **UCCSD Pool** (`uccsd_pool`): Spin-conserving UCCSD operators with singles and doubles
- **Qubit Pool** (`qubit_pool`): Qubit-ADAPT operators with Z-only strings removed
- **Qubit Excitation Pool** (`qubit_excitation`): Qubit excitation operators using Q and Q^dagger operators

### 2. Hartree-Fock State and Energy
- **Occupation number generation** for various molecules
- **Reference state creation** using Jordan-Wigner transformation
- **HF statevector generation** and validation
- **HF energy calculation** and comparison with exact energies
- **State properties validation** (normalization, particle conservation, etc.)

## Test Functions

### Helper Functions

#### `analyze_pauli_patterns(operator_pool, pool_name)`
- Analyzes Pauli string patterns in an operator pool
- Returns detailed statistics about pattern types, lengths, and operator counts
- Used by multiple tests for consistent pattern analysis

### `test_uccsd_pool()`
- Verifies UCCSD pool generation for H2-like systems (4 qubits, 2 electrons)
- Checks expected operator count (5 operators: 4 singles + 1 double)
- Validates operator types and structure
- Ensures all operators are valid QubitOperator instances

### `test_qubit_pool()`
- Tests qubit pool generation
- Verifies Z-only strings are removed
- Ensures operators contain X and/or Y operators
- **NEW**: Validates specific Pauli string patterns (XYYY, YXXX, XY, etc.)
- **NEW**: Analyzes pattern distribution by qubit count
- **NEW**: Verifies complex mixed patterns characteristic of qubit pool
- **NEW**: Compares with UCCSD pool to confirm Z operator removal
- **NEW**: Provides detailed pattern analysis and statistics

### `test_qubit_excitation_pool()`
- Tests qubit excitation pool generation
- Verifies expected operator count
- Ensures operators contain X and Y operators
- Validates Q and Q^dagger operator structure

### `test_pool_scaling()`
- Tests pool generation for different system sizes
- Verifies reasonable scaling behavior
- Tests H2-like, LiH-like, and H4-like systems

### `test_pool_consistency()`
- Ensures pools are identical across multiple calls
- Tests deterministic behavior
- Validates operator term counts

### `test_invalid_inputs()`
- Tests error handling for invalid inputs
- Verifies graceful failure for invalid pool types
- Tests boundary conditions for qubit/electron counts

### `test_operator_properties()`
- Validates operator mathematical properties
- Ensures all coefficients are finite
- Verifies valid qubit indices and Pauli types

## Hartree-Fock Test Functions

### `test_hartree_fock()`
- Main test function that runs all HF tests for multiple molecules
- Tests H4, LiH, and BeH2 molecules with different qubit/electron counts
- Validates occupation numbers, state creation, energy calculation, and state properties

### `test_occupation_numbers(mol, n_qubits, n_electrons)`
- Tests occupation number generation for HF states
- Verifies correct number of occupied and virtual orbitals
- Ensures proper ordering (occupied first, then virtual)

### `test_reference_state_creation(mol, n_qubits, n_electrons)`
- Tests reference state creation from occupation numbers
- Validates state normalization and structure
- Verifies correct Jordan-Wigner mapping

### `test_hf_statevector_creation(mol, n_qubits, n_electrons)`
- Tests HF statevector creation using Qiskit
- Validates statevector format and normalization
- Ensures correct computational basis state structure

### `test_hf_energy_calculation(mol, n_qubits, n_electrons, energy_tol)`
- Tests HF energy calculation for various molecules
- Compares HF energy with exact ground state energy
- Validates variational principle (HF energy ≥ exact energy)

### `test_state_properties(mol, n_qubits, n_electrons)`
- Tests various properties of HF states
- Validates normalization, reality, and particle conservation
- Ensures correct computational basis representation

### `test_edge_cases()`
- Tests edge cases and error handling
- Tests minimum system (H2-like)
- Validates invalid input handling

## Running the Tests

### Option 1: Using pytest (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest test_operator_pool.py -v

# Run specific test
pytest test_operator_pool.py::test_uccsd_pool -v
```

### Option 2: Using the test runner
```bash
# Make executable
chmod +x run_tests.py

# Run all tests
python run_tests.py
```

### Option 3: Direct execution
```bash
# Run the test file directly
python test_operator_pool.py
```

## Expected Results

### Operator Pool Tests
For a 4-qubit, 2-electron system (H2-like):
- **UCCSD Pool**: 5 operators (4 singles + 1 double)
- **Qubit Pool**: Similar count with Z-only strings removed
- **Qubit Excitation Pool**: 5 operators (4 singles + 1 double)

### Hartree-Fock Tests
For various molecules:
- **H4**: 8 qubits, 4 electrons - HF energy should be close to exact energy
- **LiH**: 12 qubits, 4 electrons - HF energy should be close to exact energy  
- **BeH2**: 14 qubits, 6 electrons - HF energy should be higher than exact energy
- **State Properties**: All states should be normalized, real, and conserve particle number

## Dependencies

- `pytest`: Testing framework
- `numpy`: Numerical operations
- `openfermion`: Quantum chemistry operators
- `qiskit`: Quantum computing framework
- `qiskit-aer`: Qiskit aerodynamics simulator
- `scipy`: Scientific computing library

## Troubleshooting

If tests fail:
1. Check that all dependencies are installed
2. Verify the parent directory is in the Python path
3. Check that the operator pool modules are accessible
4. Ensure the test system has sufficient memory for larger pools

## Adding New Tests

To add new tests:
1. Create a new test function with `test_` prefix
2. Add appropriate assertions
3. Include descriptive error messages
4. Add the test to the main execution block if needed
