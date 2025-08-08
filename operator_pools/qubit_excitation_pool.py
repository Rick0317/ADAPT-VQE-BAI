from openfermion import QubitOperator
import numpy as np
from itertools import combinations
from openfermion import FermionOperator, QubitOperator, hermitian_conjugated, normal_ordered, jordan_wigner, get_sparse_operator
from utils.qubit_utils import remove_z_string

def create_q_operators(n_qubits):
    """
    Create Q and Q^dagger operators for all qubits
    Q_i = (X_i + iY_i)/2 (annihilation-like)
    Q_i^dagger = (X_i - iY_i)/2 (creation-like)

    :param n_qubits: Number of qubits
    :return: (q_ops, q_dag_ops) - lists of QubitOperator objects
    """
    q_ops = []
    q_dag_ops = []

    for i in range(n_qubits):
        # Q_i = (X_i + iY_i)/2
        q_i = QubitOperator(f'X{i}', 0.5) + QubitOperator(f'Y{i}', 0.5j)
        q_ops.append(q_i)

        # Q_i^dagger = (X_i - iY_i)/2
        q_i_dag = QubitOperator(f'X{i}', 0.5) + QubitOperator(f'Y{i}', -0.5j)
        q_dag_ops.append(q_i_dag)

    return q_ops, q_dag_ops


def generate_single_excitation_generators(n_qubits, n_electrons, q_ops,
                                          q_dag_ops):
    """
    Generate single excitation generators: T_ai = Q_a^dagger * Q_i - Q_i^dagger * Q_a
    where a is virtual (unoccupied), i is occupied

    :param n_qubits: Number of qubits (orbitals)
    :param n_electrons: Number of electrons
    :param q_ops: List of Q operators
    :param q_dag_ops: List of Q^dagger operators
    :return: List of QubitOperator generators
    """
    single_generators = []

    # Occupied orbitals: 0 to n_electrons-1
    # Virtual orbitals: n_electrons to n_qubits-1
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))

    for p in range(0, n_qubits):

        for q in range(p + 1, n_qubits):

            if (p + q) % 2 == 0:
                f_operator = FermionOperator(((p, 1), (q, 0)))
                f_operator -= hermitian_conjugated(f_operator)
                f_operator = normal_ordered(f_operator)

                q_operator = remove_z_string(f_operator)
                single_generators.append(q_operator)

    # for a in virtual:  # virtual orbital (unoccupied)
    #     for i in occupied:  # occupied orbital
    #         # Check spin symmetry: only allow same-spin excitations or spin-flips
    #         a_spin = a % 2  # 0 for even (alpha), 1 for odd (beta)
    #         i_spin = i % 2  # 0 for even (alpha), 1 for odd (beta)
    #
    #         # Allow same-spin excitations and spin-flip excitations
    #         if True:  # For now, allow all excitations - can be restricted later
    #             # T_ai = Q_a^dagger * Q_i - Q_i^dagger * Q_a
    #             term1 = q_dag_ops[a] * q_ops[
    #                 i]  # Create in virtual, annihilate in occupied
    #             term2 = q_dag_ops[i] * q_ops[a]  # Hermitian conjugate
    #             generator = term1 - term2
    #
    #             # Only keep non-zero generators
    #             if generator.terms:  # Check if dictionary is non-empty
    #                 spin_type = "same-spin" if a_spin == i_spin else "spin-flip"
    #                 single_generators.append({
    #                     'operator': generator,
    #                     'type': 'single_excitation',
    #                     'indices': (a, i),  # (virtual, occupied)
    #                     'spin_type': spin_type,
    #                     'description': f'T_{a}{i}: Q_{a}^dag * Q_{i} - Q_{i}^dag * Q_{a} ({spin_type})'
    #                 })

    return single_generators


def generate_double_excitation_generators(n_qubits, n_electrons, q_ops,
                                          q_dag_ops, max_doubles=None):
    """
    Generate double excitation generators: T_abij = Q_a^dagger * Q_b^dagger * Q_i * Q_j - Q_i^dagger * Q_j^dagger * Q_a * Q_b
    where a,b are virtual (unoccupied), i,j are occupied

    :param n_qubits: Number of qubits (orbitals)
    :param n_electrons: Number of electrons
    :param q_ops: List of Q operators
    :param q_dag_ops: List of Q^dagger operators
    :param max_doubles: Maximum number of double excitations to generate (None for all)
    :return: List of QubitOperator generators
    """
    double_generators = []
    count = 0

    # Occupied orbitals: 0 to n_electrons-1
    # Virtual orbitals: n_electrons to n_qubits-1
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))




    # Generate combinations of virtual pairs (a,b) and occupied pairs (i,j)
    for a in virtual:
        for b in virtual:
            if a >= b:  # Avoid double counting and same-index pairs
                continue
            for i in occupied:
                for j in occupied:
                    if i >= j:  # Avoid double counting and same-index pairs
                        continue

                    if max_doubles and count >= max_doubles:
                        return double_generators

                    # Check spin conservation
                    # For now, allow all combinations - can add more restrictions later
                    a_spin, b_spin = a % 2, b % 2
                    i_spin, j_spin = i % 2, j % 2

                    # Determine excitation type based on spin patterns
                    if (a_spin, b_spin) == (i_spin, j_spin):
                        spin_type = "same-spin-pair"
                    elif set([a_spin, b_spin]) == set([i_spin, j_spin]):
                        spin_type = "mixed-spin"
                    else:
                        spin_type = "other"

                    # T_abij = Q_a^dagger * Q_b^dagger * Q_i * Q_j - Q_i^dagger * Q_j^dagger * Q_a * Q_b
                    term1 = q_dag_ops[a] * q_dag_ops[b] * q_ops[i] * q_ops[
                        j]  # Create in virtual, annihilate in occupied
                    term2 = q_dag_ops[i] * q_dag_ops[j] * q_ops[a] * q_ops[
                        b]  # Hermitian conjugate
                    generator = term1 - term2

                    # Only keep non-zero generators
                    if generator.terms:  # Check if dictionary is non-empty
                        double_generators.append({
                            'operator': generator,
                            'type': 'double_excitation',
                            'indices': (a, b, i, j),
                            # (virtual1, virtual2, occupied1, occupied2)
                            'spin_type': spin_type,
                            'description': f'T_{a}{b}{i}{j}: Q_{a}^dag * Q_{b}^dag * Q_{i} * Q_{j} - Q_{i}^dag * Q_{j}^dag * Q_{a} * Q_{b} ({spin_type})'
                        })
                        count += 1

    return double_generators


def generate_correct_qubit_excitation_pool(n_qubits, n_electrons,
                                           include_singles=True,
                                           include_doubles=True,
                                           max_doubles=None):
    """
    Generate the correct qubit excitation pool based on Q and Q^dagger operators

    :param n_qubits: Number of qubits (orbitals)
    :param n_electrons: Number of electrons
    :param include_singles: Whether to include single excitations
    :param include_doubles: Whether to include double excitations
    :param max_doubles: Maximum number of double excitations (None for all)
    :return: List of generator dictionaries
    """
    print(
        f"Generating correct qubit excitation pool for {n_qubits} qubits, {n_electrons} electrons...")
    print(
        f"Occupied orbitals: 0-{n_electrons - 1}, Virtual orbitals: {n_electrons}-{n_qubits - 1}")

    # Create Q and Q^dagger operators
    q_ops, q_dag_ops = create_q_operators(n_qubits)

    pool = []

    if include_singles:
        print("Generating single excitation generators...")
        singles = generate_single_excitation_generators(n_qubits, n_electrons,
                                                        q_ops, q_dag_ops)
        pool.extend(singles)
        print(f"Generated {len(singles)} single excitation generators")

    if include_doubles:
        print("Generating double excitation generators...")
        doubles = generate_double_excitation_generators(n_qubits, n_electrons,
                                                        q_ops, q_dag_ops,
                                                        max_doubles)
        pool.extend(doubles)
        print(f"Generated {len(doubles)} double excitation generators")

    # Add metadata
    for i, op_dict in enumerate(pool):
        op_dict['pool_index'] = i
        op_dict['pool_type'] = 'qubit_excitation'

    print(f"Total generators in pool: {len(pool)}")
    return pool


def get_correct_qubit_excitation_operators_only(n_qubits, n_electrons,
                                                include_singles=True,
                                                include_doubles=True,
                                                max_doubles=None):
    """
    Simplified version that returns only the QubitOperator objects

    :param n_qubits: Number of qubits (orbitals)
    :param n_electrons: Number of electrons
    :param include_singles: Whether to include single excitations
    :param include_doubles: Whether to include double excitations
    :param max_doubles: Maximum number of double excitations
    :return: List of QubitOperator objects
    """
    pool = generate_correct_qubit_excitation_pool(n_qubits, n_electrons,
                                                  include_singles,
                                                  include_doubles, max_doubles)
    return [item['operator'] for item in pool]


def analyze_correct_qubit_pool(pool):
    """
    Analyze the composition of the correct qubit excitation pool
    """
    if not pool:
        print("Empty pool!")
        return

    print(f"\n=== CORRECT QUBIT EXCITATION POOL ANALYSIS ===")
    print(f"Total generators: {len(pool)}")

    # Count by type
    type_counts = {}
    spin_type_counts = {}
    for op_dict in pool:
        op_type = op_dict['type']
        type_counts[op_type] = type_counts.get(op_type, 0) + 1

        if 'spin_type' in op_dict:
            spin_type = op_dict['spin_type']
            spin_type_counts[spin_type] = spin_type_counts.get(spin_type, 0) + 1

    print(f"\nGenerator types:")
    for op_type, count in sorted(type_counts.items()):
        print(f"  {op_type}: {count}")

    if spin_type_counts:
        print(f"\nSpin types:")
        for spin_type, count in sorted(spin_type_counts.items()):
            print(f"  {spin_type}: {count}")

    # Analyze term complexity
    term_counts = {}
    for op_dict in pool:
        num_terms = len(op_dict['operator'].terms)
        term_counts[num_terms] = term_counts.get(num_terms, 0) + 1

    print(f"\nNumber of Pauli terms per generator:")
    for num_terms, count in sorted(term_counts.items()):
        print(f"  {num_terms} terms: {count} generators")

    # Show examples
    print(f"\nExample generators:")
    for i, op_dict in enumerate(pool[:3]):
        print(f"  {i + 1}. {op_dict['description']}")
        print(f"     Terms: {len(op_dict['operator'].terms)}")
        # Show first few terms
        terms = list(op_dict['operator'].terms.items())
        for j, (pauli_string, coeff) in enumerate(terms[:3]):
            pauli_str = ' '.join([f"{op}{qubit}" for qubit, op in
                                  pauli_string]) if pauli_string else "I"
            print(f"       {coeff:.3f} * {pauli_str}")
        if len(terms) > 3:
            print(f"       ... and {len(terms) - 3} more terms")
        print()


if __name__ == "__main__":
    print("Testing correct qubit excitation pool generation...")

    # Test with small system (H2: 4 qubits, 2 electrons)
    n_qubits = 4
    n_electrons = 2
    print(f"\n{'=' * 60}")
    print(f"Testing with {n_qubits} qubits, {n_electrons} electrons (like H2)")

    # Generate pool with both singles and doubles (limited)
    pool = generate_correct_qubit_excitation_pool(
        n_qubits, n_electrons,
        include_singles=True,
        include_doubles=True,
        max_doubles=20
    )

    analyze_correct_qubit_pool(pool)

    # Test with larger system, singles only (6 qubits, 4 electrons)
    print(f"\n{'=' * 60}")
    print("Testing singles only with 6 qubits, 4 electrons")

    pool_singles = generate_correct_qubit_excitation_pool(
        6, 4,
        include_singles=True,
        include_doubles=False
    )

    analyze_correct_qubit_pool(pool_singles)

    # Test with H4 system (8 qubits, 4 electrons)
    print(f"\n{'=' * 60}")
    print("Testing H4 system: 8 qubits, 4 electrons")

    pool_h4 = generate_correct_qubit_excitation_pool(
        8, 4,
        include_singles=True,
        include_doubles=True,
        max_doubles=50
    )

    analyze_correct_qubit_pool(pool_h4)
