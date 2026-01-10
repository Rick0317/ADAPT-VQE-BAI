"""
The Generalized Fermionic Pool
"""
from openfermion import FermionOperator, jordan_wigner, normal_ordered, bravyi_kitaev
import numpy as np

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

def normalize_op(operator):
    """
    Normalize Qubit or Fermion Operator by forcing the absolute values of the coefficients to sum to zero.
    This function modifies the operator.

    Arguments:
        operator (Union[FermionOperator,QubitOperator]): the operator to normalize

    Returns:
        operator (Union[FermionOperator,QubitOperator]): the same operator, now normalized0
    """

    if operator:
        coeff = 0
        for t in operator.terms:
            coeff_t = operator.terms[t]
            # coeff += np.abs(coeff_t * coeff_t)
            coeff += np.abs(coeff_t)

        # operator = operator/np.sqrt(coeff)
        operator = operator / coeff

    return operator


def get_spin_considered_uccsd_anti_hermitian(site: int, n_elec:int):
    """
    quartic size of the generator pool.
    :param site:
    :param n_elec:
    :return:
    """
    operator_pool = []
    for p in range(0, n_elec):
        for q in range(n_elec, site):
            if (p + q) % 2 == 0:
                operator_pool.append(jordan_wigner(get_anti_hermitian_one_body((p, q))))
    for p in range(0, n_elec):
        for q in range(p, n_elec):
            for r in range(n_elec, site - 1):
                for s in range(r + 1, site):
                    if (p + q + r + s) % 2 != 0:
                        continue
                    if p % 2 + q % 2 != r % 2 + s % 2:
                        continue
                    operator_pool.append(jordan_wigner(get_anti_hermitian_two_body((p, q, r, s))))

    return operator_pool



def get_spin_considered_uccsd_jordan_wigner(site: int, n_elec:int):
    """
    quartic size of the generator pool.
    :param site:
    :param n_elec:
    :return:
    """
    operator_pool = []
    for p in range(0, n_elec):
        for q in range(n_elec, site):
            if (p + q) % 2 == 0:
                operator_pool.append(jordan_wigner(get_anti_hermitian_one_body((p, q))))
    for p in range(0, n_elec):
        for q in range(p + 1, n_elec):
            for r in range(n_elec, site - 1):
                for s in range(r + 1, site):
                    if (p + q + r + s) % 2 != 0:
                        continue
                    if p % 2 + q % 2 != r % 2 + s % 2:
                        continue
                    operator_pool.append(jordan_wigner(get_anti_hermitian_two_body((p, q, r, s))))

    return operator_pool


def get_tapered_fermion_uccsd_anti_hermitian(site: int, n_elec:int, tapered_site:list):
    """
    quartic size of the generator pool.
    :param site:
    :param n_elec:
    :return:
    """
    operator_pool = []
    for p in range(n_elec, site):
        for q in range(n_elec):
            if p % 2 == q % 2 and p not in tapered_site and q not in tapered_site:
                operator_pool.append(jordan_wigner(normal_ordered(get_anti_hermitian_one_body((p, q)))))


            for r in range(n_elec, site):
                for s in range(n_elec):
                    if (p % 2 == q % 2 and r % 2 == s % 2) or (p % 2 == s % 2 and r % 2 == q % 2) and all(x not in tapered_site for x in [p, q, r, s]):
                        operator_pool.append(jordan_wigner(normal_ordered(get_anti_hermitian_two_body((p, r, s, q)))))

    return operator_pool


def get_generalized_fermionic_pool(site: int, spin_conserving: bool = True):
    """
    Generate a generalized fermionic pool that includes all possible one-body and two-body
    excitation operators between any arbitrary orbitals (not restricted to occupied→virtual).

    This generalizes UCCSD by including excitations that would immediately annihilate the HF state,
    such as occupied→occupied, virtual→virtual, etc.

    The cluster operators are generalized: τ̂ᵖᵍʳˢ where p, q, r, s refer to any arbitrary orbital.

    :param site: Total number of spin orbitals
    :param spin_conserving: If True, only include spin-conserving excitations (default: True)
    :return: List of QubitOperators (Jordan-Wigner transformed)
    """
    operator_pool = []

    # One-body excitations: all possible p→q where p ≠ q
    for p in range(site):
        for q in range(site):
            if p == q:
                continue
            # Skip if spin is not conserved and we require spin conservation
            if spin_conserving and (p % 2 != q % 2):
                continue
            operator_pool.append(jordan_wigner(get_anti_hermitian_one_body((p, q))))

    # Two-body excitations: all possible (p,q)→(r,s)
    # Use ordering constraints to avoid duplicate operators
    for p in range(site):
        for q in range(p, site):
            for r in range(site):
                for s in range(r + 1, site):
                    # Skip trivial cases
                    if (p == r and q == s):
                        continue

                    # Spin conservation: total spin before = total spin after
                    if spin_conserving:
                        # Check parity conservation
                        if (p + q + r + s) % 2 != 0:
                            continue
                        # Check spin conservation: number of alpha/beta electrons conserved
                        if p % 2 + q % 2 != r % 2 + s % 2:
                            continue

                    operator_pool.append(jordan_wigner(get_anti_hermitian_two_body((p, q, r, s))))

    return operator_pool

if __name__ == '__main__':
    pool = get_spin_considered_uccsd_anti_hermitian(4, 2)
    print(pool)
