import unittest
from src.models.Specs.ModelSpec2 import ModelSpec2

class TestModelSpec2(unittest.TestCase):
    def setUp(self):
        self.model_spec = ModelSpec2(num_intermediate_layers=2)

    def test_generate_specifications(self):
        self.model_spec.generate_specifications(num_cascades=3, num_regulations=5, random_seed=42)
        print("Regulations:")
        for reg in self.model_spec.regulations:
            print(f"{reg.from_specie} regulates {reg.to_specie} with type {reg.reg_type}")
        self.assertGreater(len(self.model_spec.receptors), 0)
        self.assertGreater(len(self.model_spec.intermediate_layers), 0)
        self.assertGreater(len(self.model_spec.regulations), 0)

    def test_get_all_species(self):
        self.model_spec.generate_specifications(num_cascades=3, num_regulations=5, random_seed=42)
        all_species = (
            self.model_spec.receptors +
            [species for layer in self.model_spec.intermediate_layers for species in layer] +
            self.model_spec.outcomes
        )
        self.assertGreater(len(all_species), 0)
        self.assertIn('R1', all_species)
        # No need to test 'O' specie

if __name__ == '__main__':
    unittest.main()
