#!/usr/bin/env python3
"""
Generate Fermionic Operator representation of LiH Hamiltonian using openfermionpyscf

This script creates a LiH molecule (lithium hydride) and generates
its fermionic Hamiltonian representation using the openfermionpyscf package.
"""

import numpy as np
from openfermion import MolecularData, FermionOperator
from openfermionpyscf import run_pyscf, generate_molecular_hamiltonian
import pickle


def create_lih_molecule(r=1.0):
    """
    Create a LiH molecule (lithium hydride).

    Args:
        r (float): Interatomic distance in angstroms (default: 1.0)

    Returns:
        MolecularData: The LiH molecule data with computed properties
    """
    # Diatomic geometry: Li and H atoms along z-axis
    geometry = [
        ('Li', (0, 0, 0)),      # Lithium atom at origin
        ('H', (0, 0, r)),       # Hydrogen atom at distance r
    ]

    basis = 'sto-3g'  # Minimal basis set
    multiplicity = 1  # Even number of electrons (Li: 3e, H: 1e, total: 4e, singlet)
    charge = 0  # Neutral molecule
    description = 'LiH_diatomic'

    # Create MolecularData object
    lih_mol = MolecularData(geometry, basis, multiplicity, charge,
                           description=description)

    # Run PySCF calculations
    print(
        f"Running PySCF calculations for LiH molecule with interatomic distance {r} Å...")
    lih_mol = run_pyscf(lih_mol, run_fci=True, run_ccsd=True)

    return lih_mol

def get_lih_fermionic_hamiltonian(molecule):
    """
    Extract the fermionic Hamiltonian from a MolecularData object.

    Args:
        molecule (MolecularData): The molecular data object

    Returns:
        FermionOperator: The fermionic Hamiltonian
    """
    print("Extracting fermionic Hamiltonian...")
    fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
    return fermionic_hamiltonian


def analyze_fermionic_hamiltonian(fermionic_hamiltonian):
    """
    Analyze the structure of the fermionic Hamiltonian.

    Args:
        fermionic_hamiltonian (FermionOperator): The fermionic Hamiltonian to analyze
    """
    print("\n" + "="*60)
    print("LiH FERMIONIC HAMILTONIAN ANALYSIS")
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


def save_fermionic_hamiltonian(fermionic_hamiltonian, filename="lih_fermionic_hamiltonian.txt"):
    """
    Save the fermionic Hamiltonian to a text file.

    Args:
        fermionic_hamiltonian (FermionOperator): The fermionic Hamiltonian to save
        filename (str): Output filename
    """
    print(f"\nSaving fermionic Hamiltonian to {filename}...")

    with open(filename, 'w') as f:
        f.write("LiH Fermionic Hamiltonian\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Total terms: {len(fermionic_hamiltonian.terms)}\n\n")

        # Sort terms by coefficient magnitude for better readability
        sorted_terms = sorted(fermionic_hamiltonian.terms.items(),
                            key=lambda x: abs(x[1]), reverse=True)

        for term, coefficient in sorted_terms:
            if len(term) == 0:
                f.write(f"Constant: {coefficient:.8f}\n")
            else:
                term_str = " ".join([f"{op}{idx}" for op, idx in term])
                f.write(f"{term_str}: {coefficient:.8f}\n")

    print(f"Fermionic Hamiltonian saved to {filename}")


def main():
    """
    Main function to generate and analyze the LiH fermionic Hamiltonian.
    """
    print("LiH Fermionic Hamiltonian Generator")
    print("=" * 40)

    # Create LiH molecule
    try:
        lih_molecule = create_lih_molecule(r=1.6)
        print(f"✓ LiH molecule created successfully")
        print(f"  - Number of qubits: {lih_molecule.n_qubits}")
        print(f"  - Number of electrons: {lih_molecule.n_electrons}")
        print(f"  - FCI energy: {lih_molecule.fci_energy:.6f} Hartree")
        print(f"  - HF energy: {lih_molecule.hf_energy:.6f} Hartree")

    except Exception as e:
        print(f"✗ Error creating LiH molecule: {e}")
        return

    # Extract fermionic Hamiltonian
    try:
        fermionic_hamiltonian = get_lih_fermionic_hamiltonian(lih_molecule)
        print(f"✓ Fermionic Hamiltonian extracted successfully")

    except Exception as e:
        print(f"✗ Error extracting fermionic Hamiltonian: {e}")
        return

    # Analyze the Hamiltonian
    analyze_fermionic_hamiltonian(fermionic_hamiltonian)

    # Save to file
    try:
        save_fermionic_hamiltonian(fermionic_hamiltonian)
        print("✓ Fermionic Hamiltonian saved successfully")

    except Exception as e:
        print(f"✗ Error saving fermionic Hamiltonian: {e}")

    print("\n" + "="*40)
    print("Generation complete!")
    print("="*40)


if __name__ == "__main__":
    r = 1.0  # LiH equilibrium bond distance
    geometry = [
        ('N', (0, 0, 0)),      # Lithium atom at origin
        ('N', (0, 0, r)),       # Hydrogen atom at distance r
    ]

    basis = 'sto-3g'  # Minimal basis set
    multiplicity = 1  # Even number of electrons (Li: 3e, H: 1e, total: 4e, singlet)
    charge = 0  # Neutral molecule
    description = 'N2_diatomic'

    # Create LiH molecule
    # lih_mol = create_lih_molecule(r=1.0)

    # Get FermionOperator for the molecular Hamiltonian
    fermion_hamiltonian = generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)

    # Save to .bin file using pickle
    with open('ham_lib/n2_fer.bin', 'wb') as f:
        pickle.dump(fermion_hamiltonian, f)

    print("LiH FermionOperator saved to ham_lib/n2_fer.bin")

