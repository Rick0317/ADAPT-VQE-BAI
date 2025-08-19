import pytest
import numpy as np
from openfermion import QubitOperator
from get_generator_pool import get_generator_pool


def test_uccsd_pool():
    """Test UCCSD pool generation and verify expected operators"""
    # Test with H2-like system (4 qubits, 2 electrons)
    n_qubits = 4
    n_electrons = 2

    generator_pool = get_generator_pool('uccsd_pool', n_qubits, n_electrons)

    # Verify pool is not empty
    assert len(generator_pool) > 0, "UCCSD pool should not be empty"

    expected_operators = 3
    assert len(generator_pool) == expected_operators, f"Expected {expected_operators} operators, got {len(generator_pool)}"

    # Verify all operators are QubitOperator instances
    for op in generator_pool:
        assert isinstance(op, QubitOperator), f"Operator {op} is not a QubitOperator"
        assert len(op.terms) > 0, f"Operator {op} has no terms"

    # Verify specific operator types exist
    single_excitations = []
    for op in generator_pool:
        for pauli_string in op.terms.keys():
            if any(qubit == 0 or qubit == 1 for qubit, _ in pauli_string):
                if any(qubit == 2 or qubit == 3 for qubit, _ in pauli_string):
                    single_excitations.append(op)
                    break

    assert len(single_excitations) >= 2, f"Expected at least 2 single excitations, found {len(single_excitations)}"

    print(f"✓ UCCSD pool test passed: {len(generator_pool)} operators generated")


def analyze_pauli_patterns(operator_pool, pool_name):
    """Helper function to analyze Pauli string patterns in an operator pool"""
    pauli_patterns = []
    for op in operator_pool:
        for pauli_string in op.terms.keys():
            # Sort by qubit index for consistent pattern representation
            sorted_pauli = sorted(pauli_string, key=lambda x: x[0])
            pattern = ''.join([pauli_type for _, pauli_type in sorted_pauli])
            pauli_patterns.append(pattern)
    
    # Remove duplicates and sort
    unique_patterns = sorted(list(set(pauli_patterns)))
    
    # Analyze pattern types
    pattern_types = {}
    for pattern in unique_patterns:
        length = len(pattern)
        if length not in pattern_types:
            pattern_types[length] = []
        pattern_types[length].append(pattern)
    
    # Count X, Y, Z operators
    total_x = sum(pattern.count('X') for pattern in unique_patterns)
    total_y = sum(pattern.count('Y') for pattern in unique_patterns)
    total_z = sum(pattern.count('Z') for pattern in unique_patterns)
    
    return {
        'patterns': unique_patterns,
        'pattern_types': pattern_types,
        'total_patterns': len(unique_patterns),
        'total_x': total_x,
        'total_y': total_y,
        'total_z': total_z
    }


def test_qubit_pool():
    """Test qubit pool generation and verify expected operators"""
    # Test with H2-like system (4 qubits, 2 electrons)
    n_qubits = 4
    n_electrons = 2

    generator_pool = get_generator_pool('qubit_pool', n_qubits, n_electrons)

    # Verify pool is not empty
    assert len(generator_pool) > 0, "Qubit pool should not be empty"

    # Qubit pool should have similar number of operators as UCCSD
    # but with Z-only strings removed
    assert len(generator_pool) > 0, "Qubit pool should not be empty"

    # Verify all operators are QubitOperator instances
    for op in generator_pool:
        assert isinstance(op, QubitOperator), f"Operator {op} is not a QubitOperator"
        assert len(op.terms) > 0, f"Operator {op} has no terms"

    # Verify no Z-only strings (qubit pool removes Z-only operators)
    for op in generator_pool:
        for pauli_string in op.terms.keys():
            # Check that not all operators in the string are Z
            pauli_types = [pauli_type for _, pauli_type in pauli_string]
            assert not all(pt == 'Z' for pt in pauli_types), f"Found Z-only string: {pauli_string}"

    # Verify operators contain X and/or Y operators
    has_xy_operators = False
    for op in generator_pool:
        for pauli_string in op.terms.keys():
            pauli_types = [pauli_type for _, pauli_type in pauli_string]
            if any(pt in ['X', 'Y'] for pt in pauli_types):
                has_xy_operators = True
                break
        if has_xy_operators:
            break

    assert has_xy_operators, "Qubit pool should contain X and/or Y operators"

    # NEW: Verify specific Pauli string patterns (XYYY, YXXX, XY, etc.)
    print(f"\nAnalyzing qubit pool Pauli string patterns...")
    
    # Use helper function to analyze patterns
    pattern_analysis = analyze_pauli_patterns(generator_pool, "qubit_pool")
    unique_patterns = pattern_analysis['patterns']
    pattern_types = pattern_analysis['pattern_types']
    
    print(f"Found {len(unique_patterns)} unique Pauli string patterns: {unique_patterns}")
    print(f"Pattern distribution by length:")
    for length in sorted(pattern_types.keys()):
        patterns = pattern_types[length]
        print(f"  {length}-qubit patterns ({len(patterns)}): {patterns}")
    
    # Verify that all patterns contain only X and Y operators (no Z)
    for pattern in unique_patterns:
        assert 'Z' not in pattern, f"Found Z operator in pattern: {pattern}"
        assert len(pattern) > 0, f"Empty pattern found"
    
    # Check for expected patterns based on qubit pool characteristics
    # Qubit pool should contain various combinations of X and Y operators
    expected_pattern_types = {
        'single_qubit': False,      # X, Y
        'two_qubit': False,         # XX, XY, YX, YY
        'three_qubit': False,       # XXX, XXY, XYY, YYY, etc.
        'four_qubit': False,        # XXXX, XXXY, XXYY, XYYY, YYYY, etc.
    }
    
    for pattern in unique_patterns:
        if len(pattern) == 1:
            expected_pattern_types['single_qubit'] = True
        elif len(pattern) == 2:
            expected_pattern_types['two_qubit'] = True
        elif len(pattern) == 3:
            expected_pattern_types['three_qubit'] = True
        elif len(pattern) == 4:
            expected_pattern_types['four_qubit'] = True
    
    # Report which pattern types were found
    for pattern_type, found in expected_pattern_types.items():
        status = "✓" if found else "✗"
        print(f"  {status} {pattern_type}: {pattern_type.replace('_', ' ').title()}")
    
    # Verify we have at least some multi-qubit patterns (qubit pool should have these)
    multi_qubit_found = any([
        expected_pattern_types['two_qubit'],
        expected_pattern_types['three_qubit'],
        expected_pattern_types['four_qubit']
    ])
    assert multi_qubit_found, "Qubit pool should contain multi-qubit Pauli string patterns"
    
    # Verify specific common patterns that should exist in qubit pool
    # These are typical patterns from UCCSD operators after Z-removal
    common_patterns = ['XY', 'YX', 'XX', 'YY']
    found_common_patterns = [p for p in common_patterns if p in unique_patterns]
    print(f"Found {len(found_common_patterns)}/{len(common_patterns)} common patterns: {found_common_patterns}")
    
    # Should have at least some common patterns
    assert len(found_common_patterns) >= 2, f"Expected at least 2 common patterns, found: {found_common_patterns}"
    
    # Check for specific complex patterns that are characteristic of qubit pool
    # These patterns come from UCCSD operators after Z-removal
    complex_patterns = []
    for pattern in unique_patterns:
        if len(pattern) >= 3:
            # Check for patterns like XYYY, YXXX, XXYY, etc.
            if pattern.count('X') >= 1 and pattern.count('Y') >= 1:
                complex_patterns.append(pattern)
    
    print(f"Found {len(complex_patterns)} complex mixed patterns: {complex_patterns}")
    
    # Should have some complex patterns (this is what makes qubit pool interesting)
    if len(unique_patterns) > 2:  # Only check if we have enough patterns
        assert len(complex_patterns) >= 1, f"Expected at least 1 complex mixed pattern, found: {complex_patterns}"
    
    # Verify pattern distribution makes sense for qubit pool
    # Qubit pool should have a good mix of different pattern lengths
    pattern_lengths = [len(p) for p in unique_patterns]
    avg_pattern_length = sum(pattern_lengths) / len(pattern_lengths) if pattern_lengths else 0
    print(f"Average pattern length: {avg_pattern_length:.2f}")
    
    # Most patterns should be 2-4 qubits (typical for UCCSD-derived operators)
    reasonable_lengths = [l for l in pattern_lengths if 2 <= l <= 4]
    assert len(reasonable_lengths) >= len(pattern_lengths) * 0.7, f"At least 70% of patterns should be 2-4 qubits, found {len(reasonable_lengths)}/{len(pattern_lengths)}"

    print(f"✓ Qubit pool test passed: {len(generator_pool)} operators generated")
    print(f"✓ Pauli string pattern validation passed: {len(unique_patterns)} unique patterns")
    
    # Verify that qubit pool is actually different from UCCSD pool (Z operators removed)
    uccsd_pool = get_generator_pool('uccsd_pool', n_qubits, n_electrons)
    
    # Count Z operators in UCCSD pool
    uccsd_z_count = 0
    for op in uccsd_pool:
        for pauli_string in op.terms.keys():
            pauli_types = [pauli_type for _, pauli_type in pauli_string]
            uccsd_z_count += pauli_types.count('Z')
    
    # Count Z operators in qubit pool (should be 0)
    qubit_z_count = 0
    for op in generator_pool:
        for pauli_string in op.terms.keys():
            pauli_types = [pauli_type for _, pauli_type in pauli_string]
            qubit_z_count += pauli_types.count('Z')
    
    print(f"UCCSD pool Z operators: {uccsd_z_count}")
    print(f"Qubit pool Z operators: {qubit_z_count}")
    
    # Qubit pool should have fewer or equal Z operators than UCCSD pool
    assert qubit_z_count <= uccsd_z_count, f"Qubit pool should not have more Z operators than UCCSD pool"
    
    # If UCCSD pool has Z operators, qubit pool should have significantly fewer
    if uccsd_z_count > 0:
        z_reduction_ratio = qubit_z_count / uccsd_z_count
        print(f"Z operator reduction ratio: {z_reduction_ratio:.2f}")
        assert z_reduction_ratio < 0.5, f"Qubit pool should reduce Z operators by at least 50%, reduction ratio: {z_reduction_ratio}"
    
    print(f"✓ Z operator removal validation passed")
    
    # Summary of qubit pool characteristics
    print(f"\n📊 Qubit Pool Summary:")
    print(f"  Total operators: {len(generator_pool)}")
    print(f"  Unique Pauli patterns: {len(unique_patterns)}")
    print(f"  X operators: {pattern_analysis['total_x']}")
    print(f"  Y operators: {pattern_analysis['total_y']}")
    print(f"  Z operators: {pattern_analysis['total_z']} (should be 0)")
    print(f"  Pattern lengths: {list(pattern_types.keys())}")
    
    # Verify final assertions about qubit pool characteristics
    assert pattern_analysis['total_z'] == 0, f"Qubit pool should have 0 Z operators, found {pattern_analysis['total_z']}"
    assert pattern_analysis['total_x'] > 0, "Qubit pool should contain X operators"
    assert pattern_analysis['total_y'] > 0, "Qubit pool should contain Y operators"
    
    print(f"✓ Qubit pool characteristics validation passed")


def test_qubit_excitation_pool():
    """Test qubit excitation pool generation and verify expected operators"""
    # Test with H2-like system (4 qubits, 2 electrons)
    n_qubits = 4
    n_electrons = 2

    generator_pool = get_generator_pool('qubit_excitation', n_qubits, n_electrons)

    # Verify pool is not empty
    assert len(generator_pool) > 0, "Qubit excitation pool should not be empty"

    # For H2 (4 qubits, 2 electrons):
    # - Singles: 2 occupied × 2 virtual = 4 operators
    # - Doubles: combinations of 2 occupied × 2 virtual = 1 operator
    # Total: 5 operators
    expected_operators = 5
    assert len(generator_pool) == expected_operators, f"Expected {expected_operators} operators, got {len(generator_pool)}"

    # Verify all operators are QubitOperator instances
    for op in generator_pool:
        assert isinstance(op, QubitOperator), f"Operator {op} is not a QubitOperator"
        assert len(op.terms) > 0, f"Operator {op} has no terms"

    # Verify operators contain X and Y operators (Q and Q^dagger operators)
    has_xy_operators = False
    for op in generator_pool:
        for pauli_string in op.terms.keys():
            pauli_types = [pauli_type for _, pauli_type in pauli_string]
            if any(pt in ['X', 'Y'] for pt in pauli_types):
                has_xy_operators = True
                break
        if has_xy_operators:
            break

    assert has_xy_operators, "Qubit excitation pool should contain X and/or Y operators"

    print(f"✓ Qubit excitation pool test passed: {len(generator_pool)} operators generated")


def test_pool_scaling():
    """Test that pool sizes scale reasonably with system size"""
    # Test with different system sizes
    test_cases = [
        (4, 2),   # H2-like
        (6, 4),   # LiH-like
        (8, 4),   # H4-like
    ]

    for n_qubits, n_electrons in test_cases:
        for pool_type in ['uccsd_pool', 'qubit_pool', 'qubit_excitation']:
            generator_pool = get_generator_pool(pool_type, n_qubits, n_electrons)

            # Pool should not be empty
            assert len(generator_pool) > 0, f"Pool {pool_type} empty for {n_qubits} qubits, {n_electrons} electrons"

            # Pool size should scale reasonably (not exponentially)
            max_expected = n_qubits * n_electrons * (n_qubits - n_electrons) * 2
            assert len(generator_pool) <= max_expected, f"Pool {pool_type} too large: {len(generator_pool)} > {max_expected}"

            print(f"✓ {pool_type} scaling test passed for {n_qubits} qubits, {n_electrons} electrons: {len(generator_pool)} operators")


def test_pool_consistency():
    """Test that pools are consistent across multiple calls"""
    n_qubits = 4
    n_electrons = 2

    for pool_type in ['uccsd_pool', 'qubit_pool', 'qubit_excitation']:
        # Generate pool twice
        pool1 = get_generator_pool(pool_type, n_qubits, n_electrons)
        pool2 = get_generator_pool(pool_type, n_qubits, n_electrons)

        # Pools should be identical
        assert len(pool1) == len(pool2), f"Pool sizes differ for {pool_type}"

        # Check that operators are the same (same number of terms)
        for op1, op2 in zip(pool1, pool2):
            assert len(op1.terms) == len(op2.terms), f"Operator term counts differ for {pool_type}"

        print(f"✓ {pool_type} consistency test passed")


def test_invalid_inputs():
    """Test that invalid inputs are handled gracefully"""
    # Test with invalid pool type
    with pytest.raises(Exception):
        get_generator_pool('invalid_pool', 4, 2)

    # Test with invalid qubit/electron counts
    with pytest.raises(Exception):
        get_generator_pool('uccsd_pool', -1, 2)

    with pytest.raises(Exception):
        get_generator_pool('uccsd_pool', 4, -1)

    with pytest.raises(Exception):
        get_generator_pool('uccsd_pool', 2, 4)  # More electrons than qubits

    print("✓ Invalid input handling test passed")


def test_operator_properties():
    """Test specific properties of generated operators"""
    n_qubits = 4
    n_electrons = 2

    for pool_type in ['uccsd_pool', 'qubit_pool', 'qubit_excitation']:
        generator_pool = get_generator_pool(pool_type, n_qubits, n_electrons)

        for op in generator_pool:
            # All operators should be QubitOperator instances
            assert isinstance(op, QubitOperator)

            # All operators should have terms
            assert len(op.terms) > 0

            # All coefficients should be finite
            for coeff in op.terms.values():
                assert np.isfinite(coeff), f"Non-finite coefficient found: {coeff}"

            # Pauli strings should have valid qubit indices
            for pauli_string in op.terms.keys():
                for qubit_idx, pauli_type in pauli_string:
                    assert 0 <= qubit_idx < n_qubits, f"Invalid qubit index: {qubit_idx}"
                    assert pauli_type in ['X', 'Y', 'Z', 'I'], f"Invalid Pauli type: {pauli_type}"

        print(f"✓ {pool_type} operator properties test passed")


if __name__ == "__main__":
    # Run all tests
    test_uccsd_pool()
    test_qubit_pool()
    test_qubit_excitation_pool()
    test_pool_scaling()
    test_pool_consistency()
    test_operator_properties()
    test_invalid_inputs()
    print("\n🎉 All operator pool tests passed!")
