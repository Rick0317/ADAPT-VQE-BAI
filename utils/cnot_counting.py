"""
CNOT counting utilities for ADAPT-VQE ansatz operators.

This module provides functions to count the number of CNOT gates required
to implement ansatz operators in quantum circuits.

The counting is based on the standard decomposition of Pauli exponentials:
exp(i * theta * P) where P is a Pauli string requires:
- CNOT ladder to compute parity: (n_active - 1) CNOTs
- Rz rotation on the last active qubit
- Reverse CNOT ladder: (n_active - 1) CNOTs
- Total: 2 * (n_active - 1) CNOTs per Pauli string
"""

from openfermion import QubitOperator


def count_cnots_from_pauli_string(pauli_string):
    """
    Count CNOTs required for a single Pauli exponential exp(i*theta*P).

    Standard decomposition:
    - CNOT ladder to compute parity: (n_active - 1) CNOTs
    - Rz rotation on last qubit
    - Reverse CNOT ladder: (n_active - 1) CNOTs
    - Total: 2 * (n_active - 1) CNOTs

    Args:
        pauli_string: tuple of (qubit_idx, pauli_type) pairs from QubitOperator

    Returns:
        int: Number of CNOTs required
    """
    if not pauli_string:
        return 0

    n_active = len(pauli_string)
    if n_active <= 1:
        return 0

    return 2 * (n_active - 1)


def count_cnots_from_qubit_operator(qubit_op):
    """
    Count total CNOTs for a QubitOperator (assuming Trotterization with 1 step).

    Each term in the QubitOperator is a Pauli string that contributes
    2 * (n_active_qubits - 1) CNOTs.

    Args:
        qubit_op: OpenFermion QubitOperator

    Returns:
        int: Total CNOT count for implementing this operator
    """
    if not isinstance(qubit_op, QubitOperator):
        raise TypeError(f"Expected QubitOperator, got {type(qubit_op)}")

    total_cnots = 0
    for pauli_string, coeff in qubit_op.terms.items():
        if pauli_string:  # Skip identity term
            total_cnots += count_cnots_from_pauli_string(pauli_string)

    return total_cnots


def count_cnots_for_ansatz(generator_pool, selected_indices):
    """
    Count total CNOTs for a set of selected ansatz operators.

    Args:
        generator_pool: List of QubitOperator objects (the operator pool)
        selected_indices: List of indices of selected operators

    Returns:
        int: Total CNOT count for the ansatz
    """
    total = 0
    for idx in selected_indices:
        total += count_cnots_from_qubit_operator(generator_pool[idx])
    return total


def count_cnots_for_operator_list(operators):
    """
    Count total CNOTs for a list of QubitOperator objects.

    Args:
        operators: List of QubitOperator objects

    Returns:
        int: Total CNOT count
    """
    total = 0
    for op in operators:
        if isinstance(op, QubitOperator):
            total += count_cnots_from_qubit_operator(op)
    return total


def analyze_operator_cnots(qubit_op):
    """
    Analyze CNOT requirements for a QubitOperator in detail.

    Args:
        qubit_op: OpenFermion QubitOperator

    Returns:
        dict: Analysis results including:
            - total_cnots: Total CNOT count
            - n_terms: Number of Pauli terms
            - max_weight: Maximum Pauli weight (number of non-identity Paulis)
            - avg_weight: Average Pauli weight
            - cnot_breakdown: List of CNOTs per term
    """
    if not isinstance(qubit_op, QubitOperator):
        raise TypeError(f"Expected QubitOperator, got {type(qubit_op)}")

    total_cnots = 0
    n_terms = 0
    weights = []
    cnot_breakdown = []

    for pauli_string, coeff in qubit_op.terms.items():
        if pauli_string:  # Skip identity term
            n_terms += 1
            weight = len(pauli_string)
            weights.append(weight)
            term_cnots = count_cnots_from_pauli_string(pauli_string)
            cnot_breakdown.append(term_cnots)
            total_cnots += term_cnots

    return {
        'total_cnots': total_cnots,
        'n_terms': n_terms,
        'max_weight': max(weights) if weights else 0,
        'avg_weight': sum(weights) / len(weights) if weights else 0,
        'cnot_breakdown': cnot_breakdown
    }
