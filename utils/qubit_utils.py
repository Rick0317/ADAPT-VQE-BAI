from openfermion import QubitOperator, jordan_wigner
from qiskit.quantum_info import Operator
from openfermion import get_sparse_operator

def get_commutator_qubit(H_q: QubitOperator, qubit_op: QubitOperator) -> QubitOperator:
    """
    Return the normal ordered commutator of hamil_term and ferm_op.
    :param H_q:
    :param qubit_op:
    :return:
    """
    return H_q * qubit_op - qubit_op * H_q



def qubit_operator_to_qiskit_operator(qubit_op, n_qubits):
    """Convert a FermionOperator to a Qiskit Operator via bravyi_kitaev mapping"""
    sparse_matrix = get_sparse_operator(qubit_op, n_qubits)
    return Operator(sparse_matrix.toarray())


def remove_z_string(operator):
    """
    Removes the anticommutation string from Jordan-Wigner transformed excitations. This is equivalent to removing
    all Z operators.
    This function does not change the original operator.

    Args:
        operator (Union[FermionOperator, QubitOperator]): the operator in question

    Returns:
        new_operator (Union[FermionOperator, QubitOperator]): the same operator, with Pauli-Zs removed
    """

    if isinstance(operator, QubitOperator):
        qubit_operator = operator
    else:
        qubit_operator = jordan_wigner(operator)

    new_operator = QubitOperator()

    for term in qubit_operator.get_operators():

        coefficient = list(term.terms.values())[0]
        pauli_string = list(term.terms.keys())[0]

        new_pauli = QubitOperator((), coefficient)

        for qubit, operator in pauli_string:
            if operator != 'Z':
                new_pauli *= QubitOperator((qubit, operator))

        new_operator += new_pauli

    return new_operator
