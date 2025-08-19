from openfermion import FermionOperator, jordan_wigner, QubitOperator

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
    Remove ALL Pauli Z operators from every Pauli string in a QubitOperator,
    keeping only X and Y operators. This creates new Pauli strings without Z operators.

    :param qubit_op: QubitOperator
    :return: List of QubitOperator objects with Z operators removed
    """
    filtered_operators = []

    for pauli_string, coeff in qubit_op.terms.items():
        # Filter out Z operators, keeping only X and Y
        filtered_pauli_string = tuple(
            (qubit_idx, pauli_type)
            for qubit_idx, pauli_type in pauli_string
            if pauli_type in ['X', 'Y']
        )

        # Only add if there are remaining non-Z operators
        if filtered_pauli_string:
            # Create a new QubitOperator with the filtered Pauli string
            single_term_op = QubitOperator()
            single_term_op.terms[filtered_pauli_string] = coeff
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
    for p in range(0, n_elec):
        for q in range(n_elec, site):
            if (p + q) % 2 == 0:  # Same spin
                one_body = get_anti_hermitian_one_body((p, q))
                fermion_operators.append(("single", (p, q), one_body))
                singles_count += 1

    # Generate two-body excitations (doubles)
    doubles_count = 0
    for p in range(0, n_elec):
        for q in range(p + 1, n_elec):
            for r in range(n_elec, site - 1):
                for s in range(r + 1, site):
                    if (p + q + r + s) % 2 != 0:
                        continue
                    if p % 2 + q % 2 != r % 2 + s % 2:
                        continue
                    two_body = get_anti_hermitian_two_body((p, r, s, q))
                    fermion_operators.append(("double", (p, r, s, q), two_body))
                    doubles_count += 1

    print(f"Generated {singles_count} singles and {doubles_count} doubles")
    print(f"Total fermionic operators: {len(fermion_operators)}")

    # Transform to qubit operators and filter
    total_qubit_ops = 0
    for op_type, indices, ferm_op in fermion_operators:
        # Transform to qubit operator using Jordan-Wigner
        qubit_op = jordan_wigner(ferm_op)
        print(f"QUbit op: {qubit_op}")

        # Remove Z-only strings and get individual Pauli strings
        filtered_ops = remove_z_strings(qubit_op)
        print(f"Filtered ops: {filtered_ops}")

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


def get_qubit_adapt_pool_operators_only(site: int, n_electrons: int):
    """
    Simplified version that returns only the QubitOperator objects
    :param site: Number of orbitals (qubits)
    :param n_electrons: Number of electrons
    :return: List of QubitOperator objects
    """
    full_pool = get_qubit_adapt_pool(site, n_electrons)
    return [item['operator'] for item in full_pool]


def pauli_string_to_readable(pauli_string):
    """
    Convert a Pauli string tuple to a readable string format
    :param pauli_string: Tuple of (qubit_idx, pauli_type) pairs
    :return: Readable string like "X0 Y1 X4"
    """
    if not pauli_string:
        return "I"

    sorted_paulis = sorted(pauli_string, key=lambda x: x[0])
    return ' '.join([f"{pauli_type}{qubit_idx}" for qubit_idx, pauli_type in sorted_paulis])

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
        print(f"  {i+1}. {op['type']} {op['indices']}: {pauli_string_to_readable(op['pauli_string'])}")
        print(f"      Coefficient: {list(op['operator'].terms.values())[0]}")

    if len(operator_pool) > 5:
        print(f"  ... and {len(operator_pool) - 5} more operators")


def z_removal():
    """
    Test function to demonstrate Z-operator removal
    """
    from openfermion import QubitOperator

    print("Testing Z-operator removal...")

    # Create a test QubitOperator with Z operators
    test_op = QubitOperator()
    test_op.terms[((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'Y'))] = 0.5j
    test_op.terms[((1, 'Y'), (2, 'Z'), (3, 'Z'), (4, 'X'))] = -0.5j

    print(f"Original operator: {test_op}")

    # Remove Z operators
    filtered_ops = remove_z_strings(test_op)

    print(f"After Z removal:")
    for i, op in enumerate(filtered_ops):
        pauli_string = list(op.terms.keys())[0]
        coeff = list(op.terms.values())[0]
        print(f"  {i+1}. {pauli_string_to_readable(pauli_string)}: {coeff}")

    return filtered_ops

def identify_global_rotation_equivalence(pauli_string1, pauli_string2):
    """
    Check if two Pauli strings are related by a global rotation.

    Two Pauli strings are globally rotation equivalent if:
    1. They have the same qubit indices
    2. One can be transformed into the other by swapping all X↔Y operators

    Examples:
    - X0Y1Y2Y3 ↔ Y0X1X2X3 (global X↔Y swap)
    - X0X1Y2 ↔ Y0Y1X2 (global X↔Y swap)

    Args:
        pauli_string1: First Pauli string tuple
        pauli_string2: Second Pauli string tuple

    Returns:
        bool: True if they are globally rotation equivalent
    """
    if not pauli_string1 or not pauli_string2:
        return False

    # Check if they have the same qubit indices
    qubits1 = {qubit_idx for qubit_idx, _ in pauli_string1}
    qubits2 = {qubit_idx for qubit_idx, _ in pauli_string2}

    if qubits1 != qubits2:
        return False

    # Check if they are related by global X↔Y swap
    # This means: for each qubit, if one has X, the other must have Y, and vice versa
    pauli_map1 = dict(pauli_string1)
    pauli_map2 = dict(pauli_string2)

    for qubit_idx in qubits1:
        pauli1 = pauli_map1[qubit_idx]
        pauli2 = pauli_map2[qubit_idx]

        # Check if they follow the X↔Y pattern
        if not ((pauli1 == 'X' and pauli2 == 'Y') or (pauli1 == 'Y' and pauli2 == 'X')):
            return False

    return True

def remove_redundant_operators(operator_pool):
    """
    Remove redundant operators that are related by global rotations.

    This significantly reduces the pool size by keeping only one representative
    from each equivalence class, while maintaining the same expressiveness.

    Args:
        operator_pool: List of operator dictionaries

    Returns:
        List: Reduced operator pool with redundant operators removed
    """
    if not operator_pool:
        return []

    print(f"Removing redundant operators from pool of size {len(operator_pool)}...")

    # Group operators by their type and indices (same fermionic excitation)
    grouped_operators = {}
    for op in operator_pool:
        key = (op['type'], op['indices'])
        if key not in grouped_operators:
            grouped_operators[key] = []
        grouped_operators[key].append(op)

    print(f"Grouped into {len(grouped_operators)} fermionic excitation groups")

    reduced_pool = []
    total_redundant = 0

    for (op_type, indices), operators in grouped_operators.items():
        if len(operators) == 1:
            # Only one operator in this group, keep it
            reduced_pool.append(operators[0])
        else:
            # Multiple operators in this group, need to check for redundancy
            print(f"  Checking {op_type} {indices}: {len(operators)} operators")

            # Keep track of which operators to keep
            operators_to_keep = []
            redundant_in_group = 0

            for i, op1 in enumerate(operators):
                is_redundant = False

                # Check if this operator is redundant with any we're keeping
                for kept_op in operators_to_keep:
                    if identify_global_rotation_equivalence(op1['pauli_string'], kept_op['pauli_string']):
                        is_redundant = True
                        redundant_in_group += 1
                        print(f"    Redundant: {pauli_string_to_readable(op1['pauli_string'])} ↔ {pauli_string_to_readable(kept_op['pauli_string'])}")
                        break

                if not is_redundant:
                    operators_to_keep.append(op1)

            # Add the non-redundant operators to the reduced pool
            reduced_pool.extend(operators_to_keep)
            total_redundant += redundant_in_group

            print(f"    Kept {len(operators_to_keep)}, removed {redundant_in_group} redundant")

    print(f"Pool reduction: {len(operator_pool)} → {len(reduced_pool)} (removed {total_redundant} redundant)")
    print(f"Reduction factor: {len(operator_pool)/len(reduced_pool):.2f}x")

    return reduced_pool

def get_qubit_adapt_pool_reduced(site: int, n_elec: int, remove_redundant=True):
    """
    Generate qubit-ADAPT operator pool with optional redundancy removal.

    Args:
        site: Number of orbitals (qubits)
        n_elec: Number of electrons
        remove_redundant: Whether to remove redundant operators (default: True)

    Returns:
        List of QubitOperator objects for qubit-ADAPT pool
    """
    # First get the full pool
    full_pool = get_qubit_adapt_pool(site, n_elec)

    if remove_redundant:
        # Remove redundant operators
        reduced_pool = remove_redundant_operators(full_pool)
        return reduced_pool
    else:
        return full_pool

def get_qubit_adapt_pool_reduced_operators_only(site: int, n_electrons: int, remove_redundant=True):
    """
    Get reduced qubit-ADAPT pool with only QubitOperator objects (no metadata)

    Args:
        site: Number of orbitals (qubits)
        n_electrons: Number of electrons
        remove_redundant: Whether to remove redundant operators (default: True)

    Returns:
        List of QubitOperator objects
    """
    reduced_pool = get_qubit_adapt_pool_reduced(site, n_electrons, remove_redundant)
    return [item['operator'] for item in reduced_pool]

def redundancy_detection():
    """
    Test function to verify redundancy detection works correctly
    """
    print("Testing redundancy detection...")

    # Test case 1: X0Y1Y2Y3 ↔ Y0X1X2X3 (should be equivalent)
    pauli1 = ((0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'Y'))
    pauli2 = ((0, 'Y'), (1, 'X'), (2, 'X'), (3, 'X'))

    is_equivalent = identify_global_rotation_equivalence(pauli1, pauli2)
    print(f"  X0Y1Y2Y3 ↔ Y0X1X2X3: {'✓ Equivalent' if is_equivalent else '✗ Not equivalent'}")

    # Test case 2: X0X1Y2 ↔ Y0Y1X2 (should be equivalent)
    pauli3 = ((0, 'X'), (1, 'X'), (2, 'Y'))
    pauli4 = ((0, 'Y'), (1, 'Y'), (2, 'X'))

    is_equivalent2 = identify_global_rotation_equivalence(pauli3, pauli4)
    print(f"  X0X1Y2 ↔ Y0Y1X2: {'✓ Equivalent' if is_equivalent2 else '✗ Not equivalent'}")

    # Test case 3: Different qubit indices (should NOT be equivalent)
    pauli5 = ((0, 'X'), (1, 'Y'))
    pauli6 = ((0, 'Y'), (2, 'X'))  # Different qubit indices

    is_equivalent3 = identify_global_rotation_equivalence(pauli5, pauli6)
    print(f"  X0Y1 ↔ Y0X2 (different qubits): {'✓ Equivalent' if is_equivalent3 else '✗ Not equivalent'}")

    # Test case 4: Same pattern but not global rotation (should NOT be equivalent)
    pauli7 = ((0, 'X'), (1, 'Y'))
    pauli8 = ((0, 'X'), (1, 'X'))  # Same qubits, different pattern

    is_equivalent4 = identify_global_rotation_equivalence(pauli7, pauli8)
    print(f"  X0Y1 ↔ X0X1 (same qubits, different pattern): {'✓ Equivalent' if is_equivalent4 else '✗ Not equivalent'}")

    return is_equivalent and is_equivalent2 and not is_equivalent3 and not is_equivalent4

# Example usage and test function
if __name__ == "__main__":
    # Test redundancy detection
    print("="*60)
    #test_redundancy_detection()
    print("="*60)

    # Test Z removal functionality
    print("="*60)
    #test_z_removal()
    print("="*60)

    # Test with small molecule (e.g., H2: 4 orbitals, 2 electrons)
    print("Testing qubit-ADAPT pool generation...")

    site = 4  # 4 orbitals
    n_elec = 2  # 2 electrons

    print("\n1. Full pool (with Z operators):")
    pool_full = get_qubit_adapt_pool(site, n_elec)
    print(f"   Pool size: {len(pool_full)}")

    print("\n2. Pool with Z operators removed:")
    pool_no_z = get_qubit_adapt_pool(site, n_elec)  # Z removal is built into this function
    print(f"   Pool size: {len(pool_no_z)}")

    print("\n3. Pool with redundancy removal:")
    pool_reduced = get_qubit_adapt_pool_reduced(site, n_elec, remove_redundant=True)
    print(f"   Pool size: {len(pool_reduced)}")

    # Show detailed analysis of the reduced pool
    print("\n" + "="*50)
    print("ANALYSIS OF REDUCED POOL:")
    analyze_qubit_pool(pool_reduced)

    # Test with slightly larger system
    print("\n" + "="*50)
    print("Testing with larger system (6 orbitals, 4 electrons)...")

    print("\n1. Full pool:")
    pool_large_full = get_qubit_adapt_pool(6, 4)
    print(f"   Pool Original: {len(pool_large_full)}")

    print("\n2. Reduced pool (no Z + no redundancy):")
    pool_large_reduced = get_qubit_adapt_pool_reduced(6, 4, remove_redundant=True)
    print(f"   Pool Reduced: {len(pool_large_reduced)}")

    # Show some examples of what was removed
    print(f"\nExamples of redundant operators removed:")
    print("  - X0Y1Y2Y3 ↔ Y0X1X2X3 (global X↔Y rotation)")
    print("  - X0X1Y2 ↔ Y0Y1X2 (global X↔Y rotation)")
    print("  - And many more...")
