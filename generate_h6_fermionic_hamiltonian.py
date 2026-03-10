#!/usr/bin/env python3
"""
Generate Fermionic Operator representation of H6 Hamiltonian using openfermionpyscf

This script creates a linear H6 chain molecule (6 hydrogen atoms) and generates
its fermionic Hamiltonian representation using the openfermionpyscf package.

Configuration:
- Geometry: Linear chain along z-axis
- Bond length: 1.5 Angstrom between consecutive atoms
- Basis: STO-3G (minimal)
- Qubits: 12 (6 spatial orbitals x 2 spin)
- Electrons: 6
"""

import numpy as np
from openfermion import MolecularData, FermionOperator, get_fermion_operator
from openfermionpyscf import run_pyscf, generate_molecular_hamiltonian
import pickle
import os


def create_h6_molecule(bond_length=1.5):
    """
    Create a linear H6 chain molecule.

    Args:
        bond_length (float): Distance between consecutive hydrogen atoms in Angstroms (default: 1.5)

    Returns:
        MolecularData: The H6 molecule data with computed properties
    """
    # Linear geometry: 6 H atoms along z-axis with equal spacing
    geometry = [
        ('H', (0, 0, 0 * bond_length)),
        ('H', (0, 0, 1 * bond_length)),
        ('H', (0, 0, 2 * bond_length)),
        ('H', (0, 0, 3 * bond_length)),
        ('H', (0, 0, 4 * bond_length)),
        ('H', (0, 0, 5 * bond_length)),
    ]

    basis = 'sto-3g'  # Minimal basis set
    multiplicity = 1  # Singlet (6 electrons paired)
    charge = 0  # Neutral molecule
    description = f'H6_linear_r{bond_length}'

    # Create MolecularData object
    h6_mol = MolecularData(geometry, basis, multiplicity, charge,
                           description=description)

    # Run PySCF calculations
    print(f"Running PySCF calculations for H6 molecule with bond length {bond_length} Angstrom...")
    h6_mol = run_pyscf(h6_mol, run_fci=True, run_ccsd=True)

    return h6_mol


def get_h6_fermionic_hamiltonian(molecule):
    """
    Extract the fermionic Hamiltonian from a MolecularData object.

    Args:
        molecule (MolecularData): The molecular data object

    Returns:
        FermionOperator: The fermionic Hamiltonian
    """
    print("Extracting fermionic Hamiltonian...")
    # get_molecular_hamiltonian returns InteractionOperator
    # Convert to FermionOperator using get_fermion_operator
    interaction_hamiltonian = molecule.get_molecular_hamiltonian()
    fermionic_hamiltonian = get_fermion_operator(interaction_hamiltonian)
    return fermionic_hamiltonian


def analyze_fermionic_hamiltonian(fermionic_hamiltonian):
    """
    Analyze the structure of the fermionic Hamiltonian.

    Args:
        fermionic_hamiltonian (FermionOperator): The fermionic Hamiltonian to analyze
    """
    print("\n" + "="*60)
    print("H6 FERMIONIC HAMILTONIAN ANALYSIS")
    print("="*60)

    # Get basic information
    n_terms = len(fermionic_hamiltonian.terms)
    print(f"Total number of terms: {n_terms}")

    # Analyze term types
    one_body_terms = 0
    two_body_terms = 0
    constant_term = 0
    other_terms = 0

    for term, coefficient in fermionic_hamiltonian.terms.items():
        if len(term) == 0:
            constant_term += 1
        elif len(term) == 2:
            one_body_terms += 1
        elif len(term) == 4:
            two_body_terms += 1
        else:
            other_terms += 1

    print(f"Constant terms: {constant_term}")
    print(f"One-body terms: {one_body_terms}")
    print(f"Two-body terms: {two_body_terms}")
    print(f"Other terms: {other_terms}")

    # Show some example terms
    print("\nExample terms:")
    count = 0
    for term, coefficient in fermionic_hamiltonian.terms.items():
        if count < 10:  # Show first 10 terms
            print(f"  {term}: {coefficient:.6f}")
            count += 1
        else:
            break

    if n_terms > 10:
        print(f"  ... and {n_terms - 10} more terms")


def save_fermionic_hamiltonian_binary(fermionic_hamiltonian, filename="ham_lib/h6_fer.bin"):
    """
    Save the fermionic Hamiltonian to a binary file using pickle.

    Args:
        fermionic_hamiltonian (FermionOperator): The fermionic Hamiltonian to save
        filename (str): Output filename
    """
    print(f"\nSaving fermionic Hamiltonian to {filename}...")

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as f:
        pickle.dump(fermionic_hamiltonian, f)

    print(f"Fermionic Hamiltonian saved to {filename}")


def save_fermionic_hamiltonian_text(fermionic_hamiltonian, filename="h6_fermionic_hamiltonian.txt"):
    """
    Save the fermionic Hamiltonian to a text file for inspection.

    Args:
        fermionic_hamiltonian (FermionOperator): The fermionic Hamiltonian to save
        filename (str): Output filename
    """
    print(f"\nSaving human-readable Hamiltonian to {filename}...")

    with open(filename, 'w') as f:
        f.write("H6 Linear Chain Fermionic Hamiltonian\n")
        f.write("Bond length: 1.5 Angstrom\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Total terms: {len(fermionic_hamiltonian.terms)}\n\n")

        # Sort terms by coefficient magnitude for better readability
        sorted_terms = sorted(fermionic_hamiltonian.terms.items(),
                            key=lambda x: abs(x[1]), reverse=True)

        for term, coefficient in sorted_terms:
            if len(term) == 0:
                f.write(f"Constant: {coefficient:.8f}\n")
            else:
                # FermionOperator terms are tuples of (orbital_index, action)
                # where action is 1 for creation (dagger) and 0 for annihilation
                term_str = " ".join([f"{idx}^" if action == 1 else f"{idx}" for idx, action in term])
                f.write(f"{term_str}: {coefficient:.8f}\n")

    print(f"Human-readable Hamiltonian saved to {filename}")


def main(bond_length=1.5):
    """
    Main function to generate and analyze the H6 fermionic Hamiltonian.

    Args:
        bond_length (float): Bond length in Angstroms (default: 1.5)
    """
    print("H6 Linear Chain Fermionic Hamiltonian Generator")
    print("=" * 50)
    print(f"Bond length: {bond_length} Angstrom")
    print(f"Geometry: Linear chain (6 H atoms along z-axis)")

    # Create H6 molecule
    try:
        h6_molecule = create_h6_molecule(bond_length)
        print(f"\nH6 molecule created successfully")
        print(f"  - Number of qubits: {h6_molecule.n_qubits}")
        print(f"  - Number of electrons: {h6_molecule.n_electrons}")
        print(f"  - FCI energy: {h6_molecule.fci_energy:.6f} Hartree")
        print(f"  - HF energy: {h6_molecule.hf_energy:.6f} Hartree")
        print(f"  - CCSD energy: {h6_molecule.ccsd_energy:.6f} Hartree")
        print(f"  - Correlation energy: {h6_molecule.fci_energy - h6_molecule.hf_energy:.6f} Hartree")

    except Exception as e:
        print(f"Error creating H6 molecule: {e}")
        raise

    # Extract fermionic Hamiltonian
    try:
        fermionic_hamiltonian = get_h6_fermionic_hamiltonian(h6_molecule)
        print(f"\nFermionic Hamiltonian extracted successfully")

    except Exception as e:
        print(f"Error extracting fermionic Hamiltonian: {e}")
        raise

    # Analyze the Hamiltonian
    analyze_fermionic_hamiltonian(fermionic_hamiltonian)

    # Save to binary file
    try:
        save_fermionic_hamiltonian_binary(fermionic_hamiltonian)
        print("Fermionic Hamiltonian saved successfully")

    except Exception as e:
        print(f"Error saving fermionic Hamiltonian: {e}")
        raise

    print("\n" + "="*50)
    print("H6 Hamiltonian generation complete!")
    print("="*50)

    return fermionic_hamiltonian


if __name__ == "__main__":
    import sys

    # Allow bond length to be specified as command line argument
    if len(sys.argv) > 1:
        bond_length = float(sys.argv[1])
    else:
        bond_length = 1.5  # Default bond length

    # Generate using the main function
    fermion_hamiltonian = main(bond_length)

    # Also generate directly using generate_molecular_hamiltonian for consistency
    # with other molecules in the codebase
    print("\n" + "-"*50)
    print("Alternative generation using generate_molecular_hamiltonian...")

    geometry = [
        ('H', (0, 0, 0 * bond_length)),
        ('H', (0, 0, 1 * bond_length)),
        ('H', (0, 0, 2 * bond_length)),
        ('H', (0, 0, 3 * bond_length)),
        ('H', (0, 0, 4 * bond_length)),
        ('H', (0, 0, 5 * bond_length)),
    ]

    basis = 'sto-3g'
    multiplicity = 1
    charge = 0

    # Generate using the direct function (same as other molecules in codebase)
    fermion_hamiltonian_direct = generate_molecular_hamiltonian(
        geometry, basis, multiplicity, charge
    )

    # Save to .bin file using pickle
    with open('ham_lib/h6_fer.bin', 'wb') as f:
        pickle.dump(fermion_hamiltonian_direct, f)

    print("H6 FermionOperator saved to ham_lib/h6_fer.bin")
