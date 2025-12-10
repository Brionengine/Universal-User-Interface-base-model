import unittest
import logging
from quantum_mining_main import InfiniteQubitExtension

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestInfiniteQubitExtension(unittest.TestCase):
    def test_logical_qubit_scaling(self):
        """
        Test the metaphorical and literal scaling logic of the Infinite Qubit Extension.
        Verifies that P^(2^k) calculations result in the expected magnitude.
        """
        # Test Case 1: Standard Configuration
        # Physical Qubits (P) = 156 (e.g., IBM Fez)
        # Extension Layers (k) = 7
        # Expected Exponent = 2^7 = 128
        # Logical Qubits (L) = 156^128
        
        iqe = InfiniteQubitExtension(physical_qubits=156, extension_layers=7)
        
        self.assertEqual(iqe.P, 156)
        self.assertEqual(iqe.k, 7)
        self.assertEqual(iqe.exponent, 128)
        
        # Verify the magnitude string format
        effective_qubits = iqe.get_effective_qubits()
        self.assertTrue(effective_qubits.startswith("10^"))
        
        # Check if magnitude is calculated correctly (approx 10^280)
        # log10(156^128) = 128 * log10(156) ≈ 128 * 2.193 ≈ 280.7
        self.assertEqual(iqe.logical_qubits_magnitude, 280)
        
        print(f"\nTest 1 Passed: IBM Fez (156q) -> {effective_qubits} logical qubits")

    def test_custom_configuration(self):
        """Test with different physical qubit counts and layers."""
        # Physical Qubits (P) = 133 (e.g., IBM Torino)
        # Extension Layers (k) = 5 (2^5 = 32)
        iqe = InfiniteQubitExtension(physical_qubits=133, extension_layers=5)
        
        # log10(133^32) = 32 * log10(133) ≈ 32 * 2.123 ≈ 70.1
        self.assertEqual(iqe.logical_qubits_magnitude, 67) # int(32 * 2.1238) = 67
        
        print(f"Test 2 Passed: IBM Torino (133q, 5 layers) -> {iqe.get_effective_qubits()} logical qubits")

if __name__ == '__main__':
    unittest.main()

