"""
Gradient computation methods for ADAPT-VQE.

Available methods:
- standard: Full commutator computation (O(N^6+))
- anastasiou: Efficient method from arXiv:2306.03227 (O(N^5))

The Anastasiou method reduces measurement complexity by:
1. Partitioning POOL operators into 2N anchor-based commuting sets
   - YXXX operators: grouped by Y position (anchor) -> N sets
   - XYYY operators: grouped by X position (anchor) -> N sets
2. For each Hamiltonian term (pivot), measuring all pool commutators
   simultaneously within each anchor set

Scaling: O(N^4) H terms × 2N pool sets = O(N^5)
"""

from .anastasiou_gradient import (
    AnastasiouGradientComputer,
    compute_anastasiou_gradient_all_pool,
    bai_find_best_arm_anastasiou,
    partition_qubit_pool_into_anchor_sets
)

__all__ = [
    'AnastasiouGradientComputer',
    'compute_anastasiou_gradient_all_pool',
    'bai_find_best_arm_anastasiou',
    'partition_qubit_pool_into_anchor_sets'
]
