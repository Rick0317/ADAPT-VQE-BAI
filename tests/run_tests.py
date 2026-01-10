#!/usr/bin/env python3
"""
Simple test runner for operator pool tests
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_operator_pool import *
from test_hartree_fock import *

def main():
    """Run all tests"""
    print("🧪 Running operator pool tests...\n")
    
    try:
        # Run operator pool tests
        test_uccsd_pool()
        test_qubit_pool()
        test_qubit_excitation_pool()
        test_pool_scaling()
        test_pool_consistency()
        test_operator_properties()
        test_invalid_inputs()
        
        print("\n🎉 All operator pool tests passed!")
        
        print("\n🧪 Running Hartree-Fock tests...\n")
        
        # Run Hartree-Fock tests
        test_hartree_fock()
        test_edge_cases()
        test_hamiltonian_loading()
        
        print("\n🎉 All Hartree-Fock tests passed!")
        print("\n🎉 All tests completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
