import numpy as np
from entities.paulis import PauliString
from openfermion import QubitOperator, FermionOperator, bravyi_kitaev


def qwc_decomposition(Hqub):
    sorted_terms = sorted(Hqub.terms, key=lambda x: np.abs(Hqub.terms[x]), reverse=True)
    groups = []

    for term in sorted_terms:
        pauli_str = PauliString(term)  # Create PauliString from the term only.
        coeff = Hqub.terms[term]
        found_group = False

        for idx, group in enumerate(groups):
            commute = True
            for pauli_term, _ in group:  # Unpack (term, coeff) tuple, only need term here.
                group_pauli = PauliString(pauli_term)
                if not pauli_str.qubit_wise_commute(group_pauli):
                    commute = False
                    break

            if commute:
                groups[idx].append((term, coeff))
                found_group = True
                break

        if not found_group:
            groups.append([(term, coeff)])

    decomp = []
    for group in groups:
        qubit_op = QubitOperator()
        for pauli_term, coeff in group:
            qubit_op += PauliString(pauli_term).to_qubit_operator(coeff)

        decomp.append(qubit_op)

    return decomp


def qwc_decomposition_commutator(Hqub, A):
    """
    Decompose the commutator into a qubit_wise commutation while preserving the
    commutator structure [H_j, A]
    :param Hqub:
    :param A:
    :return:
    """
    sorted_terms = sorted(Hqub.terms, key=lambda x: np.abs(Hqub.terms[x]), reverse=True)
    groups = []

    for term in sorted_terms:
        commutator = QubitOperator(term) * A - A * QubitOperator(term)
        pauli_str = PauliString()  # Create PauliString from the term only.
        coeff = Hqub.terms[term]
        found_group = False

        for idx, group in enumerate(groups):
            commute = True
            for pauli_term, _ in group:  # Unpack (term, coeff) tuple, only need term here.
                group_pauli = PauliString(pauli_term)
                if not pauli_str.qubit_wise_commute(group_pauli):
                    commute = False
                    break

            if commute:
                groups[idx].append((commutator, coeff))
                found_group = True
                break

        if not found_group:
            groups.append([(commutator, coeff)])

    decomp = []
    for group in groups:
        qubit_op = QubitOperator()
        for pauli_term, coeff in group:
            qubit_op += PauliString(pauli_term).to_qubit_operator(coeff)

        decomp.append(qubit_op)

    return decomp


def abs_of_dict_value(x):
    return np.abs(x[1])


def ferm_to_qubit(H: FermionOperator):
    Hqub = bravyi_kitaev(H)
    Hqub -= Hqub.constant
    Hqub.compress()
    Hqub.terms = dict(
    sorted(Hqub.terms.items(), key=abs_of_dict_value, reverse=True))
    return Hqub
