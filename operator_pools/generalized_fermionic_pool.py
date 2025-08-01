"""
The Generalized Fermionic Pool
"""
from openfermion import FermionOperator, jordan_wigner, normal_ordered, bravyi_kitaev


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
                operator_pool.append(bravyi_kitaev(normal_ordered(get_anti_hermitian_one_body((p, q)))))
    for p in range(0, n_elec):
        for q in range(p + 1, n_elec):
            for r in range(n_elec, site - 1):
                for s in range(r + 1, site):
                    if (p + q + r + s) % 2 != 0:
                        continue
                    if p % 2 + q % 2 != r % 2 + s % 2:
                        continue
                    operator_pool.append(bravyi_kitaev(normal_ordered(get_anti_hermitian_two_body((p, q, r, s)))))

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
                operator_pool.append(bravyi_kitaev(normal_ordered(get_anti_hermitian_one_body((p, q)))))


            for r in range(n_elec, site):
                for s in range(n_elec):
                    if (p % 2 == q % 2 and r % 2 == s % 2) or (p % 2 == s % 2 and r % 2 == q % 2) and all(x not in tapered_site for x in [p, q, r, s]):
                        operator_pool.append(bravyi_kitaev(normal_ordered(get_anti_hermitian_two_body((p, r, s, q)))))

    return operator_pool

if __name__ == '__main__':
    pool = get_spin_considered_uccsd_anti_hermitian(4, 2)
    print(pool)
