import unittest
from macquerel.simulator import Simulator, Circuit

class TestSimulator(unittest.TestCase):
    def test_basic_simulation(self):
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.rz(1, 0.3)
        circuit.measure_all()

        sim = Simulator()
        result = sim.run(circuit, shots=1000)

        self.assertTrue(isinstance(result, dict))
        self.assertTrue(all(isinstance(k, int) and isinstance(v, int) for k, v in result.items()))

if __name__ == '__main__':
    unittest.main()