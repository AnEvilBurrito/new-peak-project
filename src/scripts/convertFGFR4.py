import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.utils.TxtbcToAnt import parse_initial_conditions, parse_parameters, parse_input_assignments, parse_reversible_reactions, assemble_antimony_model
with open("src\scripts\FGFR4_model_rev2a.txtbc") as f:
    model_text = f.read()

antimony_code_initial = parse_initial_conditions(model_text)

print("Antimony code initial conditions:")
for var, val in antimony_code_initial.items():
    print(f"{var} = {val}")
# print length of the dictionary
print(f"Number of initial conditions: {len(antimony_code_initial)}")

antimony_code_parameters = parse_parameters(model_text)

print("\nAntimony code parameters:")
for param, val in antimony_code_parameters.items():
    print(f"{param} = {val}")
# print length of the dictionary
print(f"Number of parameters: {len(antimony_code_parameters)}")

antimony_code_inputs = parse_input_assignments(model_text)
print("\nAntimony code input assignments:")
for assignment in antimony_code_inputs:
    print(assignment)
# print length of the list
print(f"Number of input assignments: {len(antimony_code_inputs)}")

antimony_code_reversible_reactions = parse_reversible_reactions(model_text)
print("\nAntimony code reversible reactions:")
for reaction in antimony_code_reversible_reactions:
    print(reaction)
# print length of the list
print(f"Number of reversible reactions: {len(antimony_code_reversible_reactions)}")

antimony_code_full = assemble_antimony_model(model_text)
print("\nFull Antimony model:")
print(antimony_code_full)

# Save the full Antimony model to a file
with open("src/scripts/FGFR4_model_rev2a.ant", "w") as f:
    f.write(antimony_code_full)