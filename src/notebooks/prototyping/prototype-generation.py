from models.Specs.ModelSpec4 import ModelSpec4
from models.Specs.Drug import Drug

model_spec = ModelSpec4()
new_drug = Drug(name="D", start_time=500, default_value=500)
new_drug.add_regulation("R1", "down")

model_spec.generate_specifications(3, 0)
model_spec.add_drug(new_drug)
model_spec.add_regulation("R1", "R2", "up")
model_spec.add_regulation("R3", "I1_2", "up")
model_spec.add_regulation("I1_1", "I2_2", "up")
model_spec.add_regulation("I1_2", "I2_1", "down")
model_spec.add_regulation("I1_2", "I2_3", "down")
model_spec.add_regulation("I1_3", "I2_2", "up")
model_spec.add_regulation("I2_1", "R1", "down")
model_spec.add_regulation("I2_3", "R3", "up")
