# H_5 Fermionic Hamiltonian Generator

This project generates the Fermionic Operator representation of the H_5 Hamiltonian using the `openfermionpyscf` package.

## Overview

The H_5 molecule is a linear chain of 5 hydrogen atoms. This script:
1. Creates an H_5 molecule using PySCF calculations
2. Extracts the fermionic Hamiltonian representation
3. Analyzes the structure of the Hamiltonian
4. Saves the results to a text file

## Requirements

Install the required packages:

```bash
pip install -r requirements_h5_hamiltonian.txt
```

Or install manually:

```bash
pip install openfermion openfermionpyscf pyscf numpy scipy
```

## Usage

### Basic Usage

Run the script directly:

```bash
python generate_h5_fermionic_hamiltonian.py
```

### Custom Interatomic Distance

You can modify the interatomic distance in the script by changing the `r` parameter in the `create_h5_molecule()` function call in the `main()` function.

### Programmatic Usage

```python
from generate_h5_fermionic_hamiltonian import create_h5_molecule, get_h5_fermionic_hamiltonian

# Create H_5 molecule with custom interatomic distance
h5_mol = create_h5_molecule(r=1.5)  # 1.5 Å

# Extract fermionic Hamiltonian
fermionic_hamiltonian = get_h5_fermionic_hamiltonian(h5_mol)

# Access the Hamiltonian terms
for term, coefficient in fermionic_hamiltonian.terms.items():
    print(f"Term: {term}, Coefficient: {coefficient}")
```

## Output

The script generates:

1. **Console output** showing:
   - Molecule creation progress
   - Hamiltonian extraction status
   - Analysis of the fermionic Hamiltonian structure
   - Number of terms by type (constant, one-body, two-body, etc.)
   - Example terms

2. **Text file** (`h5_fermionic_hamiltonian.txt`) containing:
   - All Hamiltonian terms sorted by coefficient magnitude
   - Term representations in fermionic operator notation

## Molecular Details

- **Geometry**: Linear chain of 5 H atoms along the z-axis
- **Basis set**: STO-3G (minimal basis)
- **Multiplicity**: 2 (odd number of electrons)
- **Charge**: 0 (neutral molecule)
- **Number of qubits**: 10 (2 orbitals per H atom)
- **Number of electrons**: 5

## Fermionic Operator Notation

The fermionic Hamiltonian is represented using creation (†) and annihilation operators:
- `0^` represents creation operator on orbital 0
- `0` represents annihilation operator on orbital 0
- Terms like `0^ 1` represent one-body interactions
- Terms like `0^ 1^ 2 3` represent two-body interactions

## Example Output

```
H_5 Fermionic Hamiltonian Generator
========================================
Running PySCF calculations for H_5 molecule with interatomic distance 1.0 Å...
✓ H_5 molecule created successfully
  - Number of qubits: 10
  - Number of electrons: 5
  - FCI energy: -2.123456 Hartree
  - HF energy: -2.100000 Hartree
✓ Fermionic Hamiltonian extracted successfully

============================================================
FERMIONIC HAMILTONIAN ANALYSIS
============================================================
Total number of terms: 100
Constant terms: 1
One-body terms: 20
Two-body terms: 79
Other terms: 0

Example terms:
  (): -2.123456
  (('0^', 0), ('0', 0)): -0.500000
  (('1^', 1), ('1', 1)): -0.500000
  ...
```

## Troubleshooting

### Common Issues

1. **PySCF installation problems**: Ensure you have a working PySCF installation
2. **Memory issues**: H_5 calculations can be memory-intensive
3. **Convergence issues**: Try different interatomic distances or basis sets

### Performance Notes

- The script runs FCI calculations which can be computationally expensive
- Consider using smaller basis sets for faster execution
- The STO-3G basis provides a good balance between accuracy and speed

## Dependencies

- **openfermion**: Core quantum chemistry library
- **openfermionpyscf**: PySCF interface for OpenFermion
- **pyscf**: Python-based quantum chemistry package
- **numpy**: Numerical computing
- **scipy**: Scientific computing

## License

This project follows the same license as the parent ADAPT-VQE-BAI project.

## Contributing

Feel free to modify the script to:
- Use different molecular geometries
- Implement different basis sets
- Add additional analysis features
- Optimize performance

