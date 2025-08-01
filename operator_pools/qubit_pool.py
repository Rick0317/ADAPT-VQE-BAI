from openfermion import FermionOperator, bravyi_kitaev, QubitOperator

def get_anti_hermitian_one_body(indices: tuple):
    """
    Get the one-body anti-Hermitian generator
    :param indices:
    :return:
    """
    p, q = indices
    return FermionOperator(f'{p}^ {q}') - FermionOperator(f'{q}^ {p}')


def get_anti_hermitian_two_body(indices: tuple):
    """
    Get the two-body anti-Hermitian generator
    :param indices:
    :return:
    """
    p, q, r, s = indices
    return FermionOperator(f'{p}^ {q}^ {r} {s}') - FermionOperator(f'{s}^ {r}^ {q} {p}')


def remove_z_strings(qubit_op):
    """
    Remove Pauli strings that contain only Z operators from a QubitOperator
    and return individual non-Z Pauli strings as separate operators
    :param qubit_op: QubitOperator
    :return: List of QubitOperator objects without Z-only strings
    """
    filtered_operators = []

    for pauli_string, coeff in qubit_op.terms.items():
        # Check if this Pauli string contains non-Z operators
        has_non_z = any(pauli[1] in ['X', 'Y'] for pauli in pauli_string)

        if has_non_z:  # Only keep strings with X or Y operators
            # Create a new QubitOperator with just this term
            single_term_op = QubitOperator()
            single_term_op.terms[pauli_string] = coeff
            filtered_operators.append(single_term_op)

    return filtered_operators


def get_qubit_adapt_pool(site: int, n_elec: int):
    """
    Generate qubit-ADAPT operator pool from spin-conserving UCCSD anti-Hermitian operators.
    Transforms fermionic operators to qubit operators using Jordan-Wigner and removes Z-only strings.

    :param site: Number of orbitals (qubits)
    :param n_elec: Number of electrons
    :return: List of QubitOperator objects for qubit-ADAPT pool
    """
    operator_pool = []
    fermion_operators = []

    print(f"Generating qubit-ADAPT pool for {site} orbitals, {n_elec} electrons...")

    # Generate one-body excitations (singles)
    singles_count = 0
    for p in range(n_elec, site):
        for q in range(n_elec):
            if p % 2 == q % 2:  # Same spin
                one_body = get_anti_hermitian_one_body((p, q))
                fermion_operators.append(("single", (p, q), one_body))
                singles_count += 1

    # Generate two-body excitations (doubles)
    doubles_count = 0
    for p in range(n_elec, site):
        for q in range(n_elec):
            for r in range(n_elec, site):
                for s in range(n_elec):
                    # Spin conservation: either same-spin pairs or opposite-spin pairs
                    same_spin_pairs = (p % 2 == q % 2 and r % 2 == s % 2)
                    opposite_spin_pairs = (p % 2 == s % 2 and r % 2 == q % 2)

                    if (same_spin_pairs or opposite_spin_pairs) and p != r and s != q:
                        two_body = get_anti_hermitian_two_body((p, r, s, q))
                        fermion_operators.append(("double", (p, r, s, q), two_body))
                        doubles_count += 1

    print(f"Generated {singles_count} singles and {doubles_count} doubles")
    print(f"Total fermionic operators: {len(fermion_operators)}")

    # Transform to qubit operators and filter
    total_qubit_ops = 0
    for op_type, indices, ferm_op in fermion_operators:
        # Transform to qubit operator using Bravyi Kitaev
        qubit_op = bravyi_kitaev(ferm_op)

        # Remove Z-only strings and get individual Pauli strings
        filtered_ops = remove_z_strings(qubit_op)

        # Add to pool with metadata
        for filtered_op in filtered_ops:
            operator_pool.append({
                'operator': filtered_op,
                'type': op_type,
                'indices': indices,
                'pauli_string': list(filtered_op.terms.keys())[0] if filtered_op.terms else None
            })
            total_qubit_ops += 1

    print(f"Final qubit-ADAPT pool size: {len(operator_pool)}")
    print(f"Average qubit operators per fermionic operator: {total_qubit_ops/len(fermion_operators):.1f}")

    return operator_pool


def get_qubit_adapt_pool_operators_only(site: int, n_elec: int):
    """
    Simplified version that returns only the QubitOperator objects
    :param site: Number of orbitals (qubits)
    :param n_elec: Number of electrons
    :return: List of QubitOperator objects
    """
    full_pool = get_qubit_adapt_pool(site, n_elec)
    return [item['operator'] for item in full_pool]


def analyze_qubit_pool(operator_pool):
    """
    Analyze the composition of the qubit operator pool
    :param operator_pool: List of operator dictionaries from get_spin_considered_uccsd_anti_hermitian
    """
    if not operator_pool:
        print("Empty operator pool!")
        return

    # Count by type
    singles = [op for op in operator_pool if op['type'] == 'single']
    doubles = [op for op in operator_pool if op['type'] == 'double']

    print(f"\n=== QUBIT POOL ANALYSIS ===")
    print(f"Total operators: {len(operator_pool)}")
    print(f"From singles: {len(singles)}")
    print(f"From doubles: {len(doubles)}")

    # Analyze Pauli string types
    pauli_types = {}
    for op in operator_pool:
        if op['pauli_string']:
            # Get the Pauli letters in the string
            pauli_letters = ''.join(sorted([pauli[1] for pauli in op['pauli_string']]))
            pauli_types[pauli_letters] = pauli_types.get(pauli_letters, 0) + 1

    print(f"\nPauli string types:")
    for pauli_type, count in sorted(pauli_types.items()):
        print(f"  {pauli_type}: {count}")

    # Show some examples
    print(f"\nExample operators:")
    for i, op in enumerate(operator_pool[:5]):
        print(op)
        pauli_str = op['pauli_string']
        if pauli_str:
            readable = ' '.join([f"{pauli[1]}{pauli[0]}" for pauli in pauli_str])
            print(f"  {i+1}. {op['type']} {op['indices']}: {readable}")


# Example usage and test function
if __name__ == "__main__":
    # Test with small molecule (e.g., H2: 4 orbitals, 2 electrons)
    print("Testing qubit-ADAPT pool generation...")

    site = 4  # 4 orbitals
    n_elec = 2  # 2 electrons

    pool = get_qubit_adapt_pool(site, n_elec)
    analyze_qubit_pool(pool)

    # Test with slightly larger system
    print("\n" + "="*50)
    print("Testing with larger system (6 orbitals, 4 electrons)...")

    pool_large = get_qubit_adapt_pool(6, 4)
    analyze_qubit_pool(pool_large)
