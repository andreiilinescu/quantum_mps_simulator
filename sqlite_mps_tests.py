import unittest
import numpy as np
from sqlite_mps import SQLITE_MPS
# Assume your simulator is imported from your module:
# from your_simulator_module import SQLITE_MPS

# Define the gates dictionary with properly shaped matrices.
gates = {
    "X": np.array([[0, 1],
                   [1, 0]]).reshape(2, 2),
    "H": (1/np.sqrt(2)) * np.array([[1,  1],
                                    [1, -1]]).reshape(2, 2),
    "Y": np.array([[0, -1j],
                   [1j, 0]]).reshape(2, 2),
    "Z": np.array([[1, 0],
                   [0, -1]]).reshape(2, 2),
    "S": np.array([[1, 0],
                   [0, 1j]]).reshape(2, 2),
    # CNOT acting on 2 qubits with qubit0 as control and qubit1 as target.
    "CNOT": np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]]).reshape(2, 2, 2, 2),
    # SWAP acting on 2 qubits.
    "SWAP": np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]]).reshape(2, 2, 2, 2)
}

class TestSQLITE_MPS(unittest.TestCase):

    def test_initial_state_one_qubit(self):
        """A 1-qubit simulator should start in the |0> state."""
        simulator = SQLITE_MPS(1, gates)
        expected = np.array([1, 0])
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected),
                        f"Expected initial state {expected}, got {statevector}")

    def test_apply_one_qubit_gate_X(self):
        """Applying the X gate on a 1-qubit system should flip |0> to |1>."""
        simulator = SQLITE_MPS(1, gates)
        simulator.apply_one_qbit_gate(0, "X")
        expected = np.array([0, 1])
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected),
                        f"After X gate, expected state {expected}, got {statevector}")

    def test_apply_one_qubit_gate_H(self):
        """Applying the H gate should create an equal superposition state."""
        simulator = SQLITE_MPS(1, gates)
        simulator.apply_one_qbit_gate(0, "H")
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"After H gate, expected state {expected}, got {statevector}")

    def test_apply_one_qubit_gate_Y(self):
        """The Y gate applied to |0> should yield [0, 1j]."""
        simulator = SQLITE_MPS(1, gates)
        simulator.apply_one_qbit_gate(0, "Y")
        expected = np.array([0, 1j])
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"After Y gate, expected state {expected}, got {statevector}")

    def test_apply_one_qubit_gate_Z(self):
        """The Z gate leaves |0> unchanged."""
        simulator = SQLITE_MPS(1, gates)
        simulator.apply_one_qbit_gate(0, "Z")
        expected = np.array([1, 0])
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected),
                        f"After Z gate, expected state {expected}, got {statevector}")

    def test_apply_one_qubit_gate_S(self):
        """
        For a 1-qubit system, prepare |1> by applying X,
        then apply the S gate. S|1> should yield i|1>.
        """
        simulator = SQLITE_MPS(1, gates)
        simulator.apply_one_qbit_gate(0, "X")  # |0> -> |1>
        simulator.apply_one_qbit_gate(0, "S")
        expected = np.array([0, 1j])
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"After X then S, expected state {expected}, got {statevector}")

    def test_sequence_HZH_equals_X(self):
        """
        Test that H Z H is equivalent to X.
        Starting from |0>, applying H, then Z, then H should yield the same state as applying X.
        """
        simulator1 = SQLITE_MPS(1, gates)
        simulator1.apply_one_qbit_gate(0, "H")
        simulator1.apply_one_qbit_gate(0, "Z")
        simulator1.apply_one_qbit_gate(0, "H")
        state_hzh = simulator1.get_statevector_np()
        
        simulator2 = SQLITE_MPS(1, gates)
        simulator2.apply_one_qbit_gate(0, "X")
        state_x = simulator2.get_statevector_np()
        
        self.assertTrue(np.allclose(state_hzh, state_x, atol=1e-8),
                        f"HZH sequence resulted in {state_hzh}, expected {state_x}")

    def test_apply_two_qubit_gate_CNOT(self):
        """
        For a 2-qubit system:
          - Apply X on qubit 0, so |00> becomes |10>
          - Then apply CNOT (control: qubit 0, target: qubit 1) to obtain |11>.
        """
        simulator = SQLITE_MPS(2, gates)
        simulator.apply_one_qbit_gate(0, "X")  # |00> -> |10>
        simulator.apply_two_qbit_gate(0, 1, "CNOT")
        expected = np.array([0, 0, 0, 1])  # |11>
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"After X then CNOT, expected state {expected}, got {statevector}")

    def test_apply_two_qubit_gate_SWAP(self):
        """
        For a 2-qubit system:
          - Apply X on qubit 0 so that |00> becomes |10>
          - Then apply SWAP to exchange qubits 0 and 1, yielding |01>.
        """
        simulator = SQLITE_MPS(2, gates)
        simulator.apply_one_qbit_gate(0, "X")  # |00> -> |10>
        simulator.apply_two_qbit_gate(0, 1, "SWAP")
        expected = np.array([0, 1, 0, 0])  # |01>
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"After X then SWAP, expected state {expected}, got {statevector}")

    def test_swap_inversion(self):
        """
        Applying the SWAP gate twice should return the system to its previous state.
        """
        simulator = SQLITE_MPS(2, gates)
        # Prepare a non-trivial state: apply X on qubit 1 to get |01>.
        simulator.apply_one_qbit_gate(1, "X")
        # Apply SWAP twice.
        simulator.apply_two_qbit_gate(0, 1, "SWAP")
        simulator.apply_two_qbit_gate(0, 1, "SWAP")
        expected = np.array([0, 1, 0, 0])  # Still |01>
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"After double SWAP, expected state {expected}, got {statevector}")

    def test_mixed_sequence_two_qubit(self):
        """
        Test a mixed sequence of gates on a 2-qubit system.
        Sequence:
          - Apply H on qubit 1.
          - Apply X on qubit 0.
          - Apply CNOT with control qubit 0 and target qubit 1.
        Expected:
          From |00>, H on qubit 1 yields (|00> + |01>)/√2.
          X on qubit 0 converts that to (|10> + |11>)/√2.
          CNOT then flips |10> to |11> and |11> to |10>, resulting in (|11> + |10>)/√2,
          which is identical to (|10> + |11>)/√2.
        """
        simulator = SQLITE_MPS(2, gates)
        simulator.apply_one_qbit_gate(1, "H")
        simulator.apply_one_qbit_gate(0, "X")
        simulator.apply_two_qbit_gate(0, 1, "CNOT")
        expected = np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)])
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"After mixed sequence, expected state {expected}, got {statevector}")


    def test_non_adjacent_two_qubit_gate(self):
        """
        For a 3-qubit system:
          - Start with |000>
          - Apply X on qubit 0 -> |100>
          - Apply CNOT with qubit 0 as control and qubit 2 as target
            (non-adjacent gate), which should flip qubit 2.
          - Expected final state: |101>
        """
        simulator = SQLITE_MPS(3, gates)
        simulator.apply_one_qbit_gate(0, "X")  # |000> becomes |100>
        simulator.apply_two_qbit_gate(0, 2, "CNOT")  # non-adjacent CNOT: control=0, target=2
        
        # For a 3-qubit state |q0, q1, q2> with q0 as most significant:
        # |101> corresponds to index 5 in the statevector.
        expected = np.zeros(2**3, dtype=complex)
        expected[5] = 1.0
        
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"Expected state {expected} but got {statevector}")
    
    def test_non_adjacent_cnot_4q(self):
        """
        In a 4-qubit system:
          - Start with |0000>
          - Apply X on qubit 1 to prepare |0 1 0 0>
          - Apply CNOT with qubit 1 (control) and qubit 3 (target).
            Since qubit 1 is 1, qubit 3 is flipped.
          - Expected final state: |0 1 0 1>, corresponding to index 5.
        """
        simulator = SQLITE_MPS(4, gates)
        simulator.apply_one_qbit_gate(1, "X")  # |0000> -> |0 1 0 0>
        simulator.apply_two_qbit_gate(1, 3, "CNOT")  # |0 1 0 0> -> |0 1 0 1>
        
        expected = np.zeros(2**4, dtype=complex)
        expected[5] = 1.0  # binary 0101 -> decimal 5
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"4-qubit non-adjacent CNOT failed. Expected {expected}, got {statevector}")

    def test_non_adjacent_swap_4q(self):
        """
        In a 4-qubit system:
          - Start with |0000>
          - Apply X on qubit 0 so that state becomes |1 0 0 0> (binary 1000, decimal 8)
          - Apply SWAP between qubit 0 and qubit 3 (non-adjacent).
          - Expected final state: |0 0 0 1> (binary 0001, decimal 1)
        """
        simulator = SQLITE_MPS(4, gates)
        simulator.apply_one_qbit_gate(0, "X")  # |0000> -> |1 0 0 0>
        simulator.apply_two_qbit_gate(0, 3, "SWAP")  # Swap q0 and q3
        
        expected = np.zeros(2**4, dtype=complex)
        expected[1] = 1.0  # binary 0001 -> decimal 1
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"4-qubit non-adjacent SWAP failed. Expected {expected}, got {statevector}")

    def test_non_adjacent_cnot_5q(self):
        """
        In a 5-qubit system:
          - Start with |00000>
          - Apply X on qubit 1 to prepare |0 1 0 0 0>
          - Apply CNOT with qubit 1 (control) and qubit 4 (target).
            Since qubit 1 is 1, qubit 4 flips.
          - Expected final state: |0 1 0 0 1>, corresponding to index 9.
        """
        simulator = SQLITE_MPS(5, gates)
        simulator.apply_one_qbit_gate(1, "X")  # |00000> -> |0 1 0 0 0>
        simulator.apply_two_qbit_gate(1, 4, "CNOT")  # |0 1 0 0 0> -> |0 1 0 0 1>
        
        expected = np.zeros(2**5, dtype=complex)
        expected[9] = 1.0  # binary 01001 -> decimal 9
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"5-qubit non-adjacent CNOT failed. Expected {expected}, got {statevector}")

    def test_non_adjacent_multiple_sequence_5q(self):
        """
        In a 5-qubit system, perform a sequence of non-adjacent operations:
          - Start with |00000>
          - Apply X on qubit 0 and on qubit 4 to prepare |1 0 0 0 1>
          - Apply SWAP between qubit 0 and qubit 3 (non-adjacent):
              |1 0 0 0 1> becomes |0 0 0 1 1>
          - Apply CNOT with qubit 3 (control) and qubit 1 (target):
              Since qubit 3 is 1, flip qubit 1.
              Final state becomes |0 1 0 1 1>, which is binary 01011 (decimal 11).
        """
        simulator = SQLITE_MPS(5, gates)
        simulator.apply_one_qbit_gate(0, "X")  # |00000> -> |1 0 0 0 0>
        simulator.apply_one_qbit_gate(4, "X")  # -> |1 0 0 0 1>
        simulator.apply_two_qbit_gate(0, 3, "SWAP")  # Swap qubit 0 and qubit 3: -> |0 0 0 1 1>
        simulator.apply_two_qbit_gate(3, 1, "CNOT")  # CNOT: control q3 is 1, flip q1: -> |0 1 0 1 1>
        
        expected = np.zeros(2**5, dtype=complex)
        expected[11] = 1.0  # binary 01011 -> decimal 11
        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"5-qubit non-adjacent multiple sequence failed. Expected {expected}, got {statevector}")
        
    def test_parametrized_circuit(self):
        data={"number_of_qubits": 3,"gates": [{"qubits": [0],"gate": "H","parameters":[]},{"qubits": [0],"gate": "RY","parameters":[0.244444444]},{"qubits": [0],"gate": "RY","parameters":[0.245444444]}]}
        simulator=SQLITE_MPS.run_circuit_json(data)

        expected = np.zeros(2**3, dtype=complex)
        expected[0]=0.5145250977855069+0j
        expected[4]=0.8574753196149815+0j 

        statevector = simulator.get_statevector_np()
        self.assertTrue(np.allclose(statevector, expected, atol=1e-8),
                        f"Expected state {expected} but got {statevector}")

        
    def test_unknown_gate_error(self):
        """
        Using a gate name that is not defined in the gates dictionary should raise an error.
        """
        simulator = SQLITE_MPS(1, gates)
        with self.assertRaises(Exception):
            simulator.apply_one_qbit_gate(0, "UNKNOWN")

if __name__ == '__main__':
    unittest.main()
