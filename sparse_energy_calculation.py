import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, eye
from scipy.sparse.linalg import expm_multiply, LinearOperator
from scipy.sparse.csgraph import connected_components
import gc
from qiskit.quantum_info import SparsePauliOp, Statevector
from utils.qubit_utils import get_commutator_qubit, qubit_operator_to_qiskit_operator
import time

def convert_operator_to_sparse(operator):
    """
    Convert various operator types to sparse matrix format.
    Since new_operator is now guaranteed to be scipy sparse, this mainly handles H_sparse_matrix.
    """
    print(f"    Converting operator type: {type(operator)}")

    if hasattr(operator, 'toarray'):
        # Already a sparse matrix
        print(f"    Operator is already sparse: {type(operator)}")
        return operator
    elif hasattr(operator, 'to_matrix'):
        # Qiskit operator (for H_sparse_matrix)
        print(f"    Converting Qiskit operator to sparse matrix")
        try:
            # Try sparse first
            result = operator.to_matrix(sparse=True)
            print(f"    Successfully converted to sparse matrix: {type(result)}")
            return result
        except (TypeError, AttributeError) as e:
            print(f"    Sparse conversion failed ({e}), falling back to dense")
            # Fallback to dense then convert to sparse
            dense_matrix = operator.to_matrix()
            print(f"    Converted to dense matrix: {type(dense_matrix)}, shape: {dense_matrix.shape}")
            return csr_matrix(dense_matrix)
    elif hasattr(operator, 'shape'):
        # Numpy array or similar
        print(f"    Converting numpy-like array to sparse: {type(operator)}, shape: {operator.shape}")
        return csr_matrix(operator)
    else:
        print(f"    Unknown operator type: {type(operator)}")
        raise ValueError(f"Unknown operator type: {type(operator)}")

def sparse_matrix_exponential_approximation(sparse_matrix, vector, param, max_iter=10, tol=1e-8):
    """
    Approximate sparse matrix exponential using Krylov subspace methods.
    This avoids converting to dense matrices for large systems.

    Args:
        sparse_matrix: Sparse matrix operator
        vector: State vector to apply exponential to (can be Statevector or numpy array)
        param: Parameter for the exponential
        max_iter: Maximum Krylov iterations (for fallback method)
        tol: Convergence tolerance

    Returns:
        Approximated exponential result as numpy array
    """
    if param == 0:
        # Convert Statevector to numpy array if needed
        if hasattr(vector, 'data'):
            return vector.data.copy()
        else:
            return vector.copy()

    # Convert Statevector to numpy array if needed
    if hasattr(vector, 'data'):
        vector_array = vector.data
    else:
        vector_array = vector

    # Ensure vector is numpy array with complex dtype
    if not isinstance(vector_array, np.ndarray):
        vector_array = np.array(vector_array, dtype=np.complex128)
    elif vector_array.dtype != np.complex128:
        vector_array = vector_array.astype(np.complex128)

    # Use scipy's expm_multiply which is optimized for sparse matrices
    try:
        # expm_multiply doesn't accept max_iter, only tol
        result = expm_multiply(param * sparse_matrix, vector_array)
        return result
    except Exception as e:
        print(f"    Warning: expm_multiply failed ({e}), falling back to iterative method")
        return iterative_sparse_exponential(sparse_matrix, vector_array, param, max_iter, tol)

def iterative_sparse_exponential(sparse_matrix, vector, param, max_iter=10, tol=1e-8):
    """
    Iterative Taylor series expansion for sparse matrix exponential.
    More memory efficient but potentially slower than expm_multiply.
    """
    result = vector.copy()
    temp_vector = vector.copy()
    factorial = 1.0

    for k in range(1, max_iter + 1):
        factorial *= k
        temp_vector = sparse_matrix @ temp_vector
        term = (param ** k) / factorial * temp_vector
        result += term

        # Check convergence
        if np.linalg.norm(term) < tol * np.linalg.norm(result):
            break

    return result

def sparse_energy_calculation(H_sparse_matrix, current_state, new_operator, new_param):
    """
    Sparse matrix energy calculation that maintains sparsity throughout.
    Avoids dense matrix conversions for better memory efficiency and performance.

    Args:
        H_sparse_matrix: Sparse Hamiltonian matrix (scipy sparse matrix)
        current_state: Current state vector (Statevector or numpy array)
        new_operator: New operator to apply (scipy sparse matrix, already in sparse format)
        new_param: Parameter for the new operator

    Returns:
        energy: Energy expectation value
        updated_state: Updated state vector (numpy array)
    """
    start_time = time.time()

    # Convert Statevector to numpy array if needed
    if hasattr(current_state, 'data'):
        state_array = current_state.data
    else:
        state_array = current_state

    # Ensure state is numpy array with complex dtype
    if not isinstance(state_array, np.ndarray):
        state_array = np.array(state_array, dtype=np.complex128)
    elif state_array.dtype != np.complex128:
        state_array = state_array.astype(np.complex128)

    # new_operator is already a scipy sparse matrix
    new_op_matrix = new_operator

    # Apply the new operator using sparse matrix exponential
    print(f"    Applying operator with parameter {new_param:.6f}")
    print(f"    Operator matrix shape: {new_op_matrix.shape}, sparsity: {1 - new_op_matrix.nnz / (new_op_matrix.shape[0] * new_op_matrix.shape[1]):.4f}")

    try:
        updated_state = sparse_matrix_exponential_approximation(
            new_op_matrix, state_array, new_param
        )
        print(f"    Operator application completed in {time.time() - start_time:.4f}s")
    except Exception as e:
        print(f"    Error in sparse matrix exponential: {e}")
        # Fallback to identity operation
        updated_state = state_array.copy()

    # Calculate energy using sparse matrix operations
    energy_start_time = time.time()

    try:
        # Use sparse matrix-vector multiplication directly
        H_sparse = convert_operator_to_sparse(H_sparse_matrix)
        temp_vector = H_sparse @ updated_state

        # Compute energy expectation value
        energy = np.real(np.vdot(updated_state, temp_vector))

        print(f"    Energy calculation completed in {time.time() - energy_start_time:.4f}s")
        print(f"    Energy: {energy:.8f}")

    except Exception as e:
        print(f"    Error in energy calculation: {e}")
        energy = float('inf')

    # Clean up intermediate objects
    del new_op_matrix, state_array
    if 'temp_vector' in locals():
        del temp_vector
    gc.collect()

    total_time = time.time() - start_time
    print(f"    Total sparse energy calculation time: {total_time:.4f}s")

    return energy, updated_state

def sparse_single_parameter_energy_optimization(H_sparse_matrix, current_state, new_operator,
                                             initial_guess=0.01, max_iter=50, tol=1e-6):
    """
    Single parameter energy optimization using sparse matrices throughout.

    Args:
        H_sparse_matrix: Sparse Hamiltonian matrix
        current_state: Current state vector
        new_operator: New operator to optimize
        initial_guess: Initial parameter guess
        max_iter: Maximum optimization iterations
        tol: Convergence tolerance

    Returns:
        optimal_param: Optimal parameter value
        optimal_energy: Optimal energy value
        final_state: Final state vector
    """
    from scipy.optimize import minimize

    def sparse_single_energy_function(param):
        """Single parameter energy function that uses sparse matrix operations"""
        # Apply the new operator with the given parameter
        _, state = sparse_energy_calculation(H_sparse_matrix, current_state, new_operator, param[0])

        # Calculate energy using sparse operations
        try:
            # Ensure H is in sparse format
            H_sparse = convert_operator_to_sparse(H_sparse_matrix)
            temp_vector = H_sparse @ state

            final_energy = np.real(np.vdot(state, temp_vector))
            del temp_vector

        except Exception as e:
            print(f"    Error in single parameter energy calculation: {e}")
            final_energy = float('inf')

        # Clean up intermediate state
        del state
        gc.collect()

        # Validate energy value
        if not np.isfinite(final_energy):
            print(f"    Warning: Non-finite energy detected: {final_energy}")
            return float('inf')

        return final_energy

    # Use scipy.minimize with L-BFGS-B method for single parameter optimization
    print(f"    Starting sparse single parameter optimization...")
    print(f"    Initial guess: {initial_guess:.6f}")

    # Print initial energy
    initial_energy = sparse_single_energy_function([initial_guess])
    print(f"    Initial energy: {initial_energy:.8f}")

    # Test energy calculation with zero parameter
    zero_energy = sparse_single_energy_function([0.0])
    print(f"    Energy with zero parameter: {zero_energy:.8f}")

    # Try different optimization methods for robustness
    methods_to_try = ['L-BFGS-B', 'BFGS', 'CG']
    best_result = None
    best_energy = float('inf')

    for method in methods_to_try:
        try:
            result = minimize(
                sparse_single_energy_function,
                [initial_guess],
                method=method,
                options={
                    'maxiter': max_iter,
                    'disp': False,
                    'gtol': tol,
                    'ftol': tol
                },
                bounds=[(-2*np.pi, 2*np.pi)] if method == 'L-BFGS-B' else None
            )

            if result.success and result.fun < best_energy:
                best_result = result
                best_energy = result.fun

        except Exception as e:
            print(f"    Method {method} failed: {e}")
            continue

    # If all methods failed, use the last result or fallback
    if best_result is None:
        print("    All optimization methods failed, using fallback")
        # Fallback to simple grid search
        param_range = np.linspace(-np.pi, np.pi, 100)
        energies = []
        for param in param_range:
            energy = sparse_single_energy_function([param])
            energies.append(energy)

        best_idx = np.argmin(energies)
        optimal_param = param_range[best_idx]
        optimal_energy = energies[best_idx]
    else:
        optimal_param = best_result.x[0]
        optimal_energy = best_result.fun

    # Calculate final state with optimal parameter
    optimal_energy, final_state = sparse_energy_calculation(H_sparse_matrix, current_state, new_operator, optimal_param)

    # Clean up
    del sparse_single_energy_function, best_result, best_energy
    gc.collect()

    print(f"    Single parameter optimization completed: param={optimal_param:.6f}, energy={optimal_energy:.8f}")

    return optimal_param, optimal_energy, final_state

def sparse_multi_parameter_energy_optimization(H_sparse_matrix, initial_state, ansatz_ops, params,
                                            max_iter=None, tol=1e-6):
    """
    Multi-parameter energy optimization using sparse matrices throughout.

    Args:
        H_sparse_matrix: Sparse Hamiltonian matrix
        initial_state: The initial state
        ansatz_ops: List of ansatz operators
        params: Current parameter values
        max_iter: Maximum optimization iterations
        tol: Convergence tolerance

    Returns:
        optimal_params: Optimal parameter values
        optimal_energy: Optimal energy value
        final_state: Final state vector
    """
    from scipy.optimize import minimize

    def sparse_energy_function(param_vector):
        """Energy function that uses sparse matrix operations"""
        if len(param_vector) != len(ansatz_ops):
            raise ValueError(f"Parameter vector length {len(param_vector)} doesn't match number of operators {len(ansatz_ops)}")

        # Apply all operators in sequence with their respective parameters
        state = initial_state.copy()

        for i, (operator, param) in enumerate(zip(ansatz_ops, param_vector)):
            _, state = sparse_energy_calculation(H_sparse_matrix, state, operator, param)

        # Calculate final energy using sparse operations
        try:
            # Ensure H is in sparse format
            H_sparse = convert_operator_to_sparse(H_sparse_matrix)
            temp_vector = H_sparse @ state

            final_energy = np.real(np.vdot(state, temp_vector))
            del temp_vector

        except Exception as e:
            print(f"    Error in final energy calculation: {e}")
            final_energy = float('inf')

        # Clean up intermediate state
        del state
        gc.collect()

        # Validate energy value
        if not np.isfinite(final_energy):
            print(f"    Warning: Non-finite energy detected: {final_energy}")
            return float('inf')

        return final_energy

    # Use scipy.minimize with L-BFGS-B method
    print(f"    Starting sparse multi-parameter optimization...")
    print(f"    Number of parameters: {len(params)}")

    # Create bounds for all parameters
    bounds = [(-2*np.pi, 2*np.pi) for _ in range(len(params))]

    # Print initial energy
    initial_energy = sparse_energy_function(params)
    print(f"    Initial energy: {initial_energy:.8f}")
    print(f"    Initial parameters: {[f'{p:.6f}' for p in params]}")

    # Test energy calculation with zero parameters
    if len(params) > 0:
        zero_params = [0.0] * len(params)
        zero_energy = sparse_energy_function(zero_params)
        print(f"    Energy with zero parameters: {zero_energy:.8f}")

    try:
        result = minimize(
            sparse_energy_function,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'disp': False,
                'gtol': tol,
            }
        )

        print(f"    Optimization completed: success={result.success}, energy={result.fun:.8f}, iterations={result.nit}")

        if result.success:
            param_changes = [f"{result.x[i]-params[i]:+.4f}" for i in range(len(params))]
            print(f"    Parameter changes: {param_changes}")
            optimal_params = result.x
            optimal_energy = result.fun
        else:
            print(f"    Optimization failed, using initial parameters")
            optimal_params = params
            optimal_energy = initial_energy

    except Exception as e:
        print(f"    Optimization failed: {e}, using initial parameters")
        optimal_params = params
        optimal_energy = initial_energy

    # Calculate final state with optimal parameters
    final_state = initial_state.copy()
    for operator, param in zip(ansatz_ops, optimal_params):
        _, final_state = sparse_energy_calculation(H_sparse_matrix, final_state, operator, param)

    # Clean up
    del sparse_energy_function
    gc.collect()

    return optimal_params, optimal_energy, final_state

def benchmark_sparse_vs_dense(n_qubits, H_sparse_matrix, current_state, new_operator, new_param):
    """
    Benchmark sparse vs dense matrix operations for different system sizes.

    Args:
        n_qubits: Number of qubits in the system
        H_sparse_matrix: Sparse Hamiltonian matrix
        current_state: Current state vector
        new_operator: New operator to apply
        new_param: Parameter for the new operator
    """
    print(f"\n=== Benchmarking for {n_qubits} qubits (system size: 2^{n_qubits} = {2**n_qubits}) ===")

    # Benchmark sparse version
    print("Testing sparse version...")
    sparse_start = time.time()
    sparse_memory_before = get_memory_usage()

    try:
        sparse_energy, sparse_state = sparse_energy_calculation(
            H_sparse_matrix, current_state, new_operator, new_param
        )
        sparse_time = time.time() - sparse_start
        sparse_memory_after = get_memory_usage()
        sparse_memory_used = sparse_memory_after - sparse_memory_before

        print(f"    Sparse version: {sparse_time:.4f}s, memory: {sparse_memory_used:+.1f} MB")

    except Exception as e:
        print(f"    Sparse version failed: {e}")
        sparse_time = float('inf')
        sparse_memory_used = 0

    # Benchmark dense version (original method)
    print("Testing dense version...")
    dense_start = time.time()
    dense_memory_before = get_memory_usage()

    try:
        # Import the original function
        from adapt_vqe_exact_estimates import fast_energy_calculation

        dense_energy, dense_state = fast_energy_calculation(
            H_sparse_matrix, current_state, new_operator, new_param
        )
        dense_time = time.time() - dense_start
        dense_memory_after = get_memory_usage()
        dense_memory_used = dense_memory_after - dense_memory_before

        print(f"    Dense version: {dense_time:.4f}s, memory: {dense_memory_used:+.1f} MB")

    except Exception as e:
        print(f"    Dense version failed: {e}")
        dense_time = float('inf')
        dense_memory_used = 0

    # Calculate speedup
    if sparse_time < float('inf') and dense_time < float('inf'):
        speedup = dense_time / sparse_time
        memory_ratio = dense_memory_used / sparse_memory_used if sparse_memory_used > 0 else float('inf')

        print(f"    Speedup: {speedup:.2f}x")
        print(f"    Memory ratio: {memory_ratio:.2f}x")

        # Check energy agreement
        energy_diff = abs(sparse_energy - dense_energy)
        print(f"    Energy difference: {energy_diff:.2e}")

        if energy_diff < 1e-6:
            print(f"    ✓ Energies agree within tolerance")
        else:
            print(f"    ⚠ Energy difference exceeds tolerance")

    print("=" * 60)

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except ImportError:
        return 0.0

if __name__ == "__main__":
    print("Sparse Energy Calculation Module")
    print("This module provides sparse matrix implementations for quantum energy calculations.")
    print("Use the functions:")
    print("  - sparse_energy_calculation(): Single operator application")
    print("  - sparse_multi_parameter_energy_optimization(): Multi-parameter optimization")
    print("  - benchmark_sparse_vs_dense(): Performance comparison")
