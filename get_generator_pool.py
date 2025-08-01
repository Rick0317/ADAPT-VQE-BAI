from operator_pools.generalized_fermionic_pool import get_spin_considered_uccsd_anti_hermitian, get_tapered_fermion_uccsd_anti_hermitian
from operator_pools.qubit_pool import get_qubit_adapt_pool_operators_only
from operator_pools.qubit_excitation_pool import get_correct_qubit_excitation_operators_only

def get_generator_pool(pool_type, n_site, n_elec, tapered_sites=[]):
    if pool_type == 'uccsd_pool':
        return get_spin_considered_uccsd_anti_hermitian(n_site, n_elec)

    if pool_type == 'tapered_uccsd_pool':
        return get_tapered_fermion_uccsd_anti_hermitian(n_site, n_elec, tapered_sites )

    if pool_type =='qubit_pool':
        return get_qubit_adapt_pool_operators_only(n_site, n_elec)

    if pool_type == 'qubit_excitation':
        return get_correct_qubit_excitation_operators_only(n_site, n_elec)


    return get_spin_considered_uccsd_anti_hermitian(n_site, n_elec)
