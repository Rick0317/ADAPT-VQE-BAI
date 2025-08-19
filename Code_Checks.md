# Code check for adapt_vqe_exact.py
This is a documentation of how we check the correctnress of the code in adapt_vqe_exact.py

## Line 825 ~ 857

ferm_to_qubit(): Converting from fermion to qubit Hamiltonian and removes the constant
~ exact_energy calculation　

=> The exact energies are compared to the literatures

get_occ_no: Checked the n_elec matches for molecules

get_reference_state(): The HF state is obtained exactly from Seonghoon Choi's Shared Pauli paper

## get_generator_pool()
The number of generators in the pool matches with https://arxiv.org/abs/2507.16879, this paper.

## Line 863 ~ 868

# Line 888 ~ 891
