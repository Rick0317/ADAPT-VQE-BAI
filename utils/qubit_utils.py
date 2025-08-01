from openfermion import QubitOperator
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

