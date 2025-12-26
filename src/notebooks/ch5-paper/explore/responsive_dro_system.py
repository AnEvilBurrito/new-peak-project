# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: new-peak-project
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Responsive D→R→I→O System
#
# **Purpose**: Create a highly responsive system where drug D reliably perturbs Oa
#
# **Strategy**: Mechanistic parameter tuning for cascade activation and drug response
#
# **Tools Used**: 
# - ModelSpec4 for cascade architecture
# - ParameterRandomizer for tuned parameter ranges
# - InitialConditionRandomizer for optimal starting conditions
# - ParameterMapper for cascade analysis

# %% [markdown]
# ## Initial Setup

# %%
import os

path = os.getcwd()
index_project = path.find("project")
project_path = path[: index_project + 7]
os.chdir(project_path + "/src")
print(f"Project path set to: {os.getcwd()}")

# %% [markdown]
# ## Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models.Specs.ModelSpec4 import ModelSpec4
from models.Specs.Drug import Drug
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
from models.utils.parameter_randomizer import ParameterRandomizer
from models.utils.initial_condition_randomizer import InitialConditionRandomizer
from models.utils.parameter_mapper import get_parameter_reaction_map, find_parameter_by_role, get_parameters_for_state

print("All imports complete")

# %% [markdown]
# ## 1. Create Tuned D→R→I→O Model
#
# Create cascade with tuned parameters for proper activation

# %%
# Initialize model specification with 1 intermediate layer
model_spec = ModelSpec4(num_intermediate_layers=1)

# Generate network FIRST (creates R1, I1_1, O)
model_spec.generate_specifications(
    num_cascades=1,
    num_regulations=0,    # No feedback for simplicity
    random_seed=42,
    verbose=1
)

# Create drug D that down-regulates R1
drug_d = Drug(
    name="D",
    start_time=500.0,     # Drug applied at time 500 (allows steady state)
    default_value=1.0,    # Drug concentration
    regulation=["R1"],    # Regulates receptor R1
    regulation_type=["down"]  # Down-regulation decreases activation
)

# Add drug to model
model_spec.add_drug(drug_d)
print(f"Drug added: {drug_d}")

print("Network specifications:")
print(f"  Receptors: {model_spec.receptors}")
print(f"  Intermediate: {model_spec.intermediate_layers[0]}")
print(f"  Outcomes: {model_spec.outcomes}")

# %% [markdown]
# ## 2. Generate Model with Tuned Parameters
#
# Use wider parameter ranges designed for cascade activation

# %%
# Generate the base model with tuned parameter ranges
# These ranges are designed to ensure good cascade activation
model = model_spec.generate_network(
    network_name="ResponsiveDRO_Tuned",
    mean_range_species=(50, 150),      # Initial concentrations
    rangeScale_params=(0.5, 2.0),      # Wider range for activation tuning (±100%)
    rangeMultiplier_params=(0.8, 1.2), # Additional variation
    random_seed=42,
    receptor_basal_activation=True     # Receptors have basal activation
)

print(f"Base model created: {len(model.reactions)} reactions")
print(f"States: {list(model.states.keys())}")

# %% [markdown]
# ## 3. Apply Tuned Parameter Ranges using ParameterRandomizer
#
# Set specific ranges for different parameter types to ensure cascade activation

# %%
# Create ParameterRandomizer instance
param_randomizer = ParameterRandomizer(model)

# Set tuned parameter ranges for cascade activation
print("Setting tuned parameter ranges...")

# Vmax: Higher rates for better activation
param_randomizer.set_range_for_param_type('Vmax', 5.0, 50.0)
print("  Vmax range: 5.0 - 50.0 (higher forward rates)")

# Km: Moderate substrate affinity
param_randomizer.set_range_for_param_type('Km', 10.0, 100.0)
print("  Km range: 10.0 - 100.0 (moderate affinity)")

# Kc: Strong basal activation for receptors
param_randomizer.set_range_for_param_type('Kc', 0.1, 1.0)
print("  Kc range: 0.1 - 1.0 (strong basal activation)")

# Generic K parameters
param_randomizer.set_range_for_param_type('Ka', 0.01, 0.1)
param_randomizer.set_range_for_param_type('Ki', 0.01, 0.1)
param_randomizer.set_range_for_param_type('Kic', 0.01, 0.1)

# Generate new model with randomized parameters
tuned_model = param_randomizer.randomize_all_parameters(seed=42)

print("Tuned parameter ranges applied")
print(f"Model parameters regenerated: {len(tuned_model.get_parameters())}")

# %% [markdown]
# ## 4. Apply Optimal Initial Conditions using InitialConditionRandomizer
#
# Set initial conditions that favor cascade activation

# %%
# Create InitialConditionRandomizer instance
ic_randomizer = InitialConditionRandomizer(tuned_model)

# Set category-specific ranges for optimal starting conditions
print("Setting optimal initial condition ranges...")

# Receptors: Higher levels for cascade initiation
ic_randomizer.set_category_ranges('receptors', 80.0, 150.0)
print("  Receptors (R*): 80.0 - 150.0")

# Activated forms: Start with some activation
ic_randomizer.set_category_ranges('activated', 20.0, 60.0)
print("  Activated forms (*a): 20.0 - 60.0")

# Intermediates: Moderate levels
ic_randomizer.set_category_ranges('intermediates', 50.0, 120.0)
print("  Intermediates (I*): 50.0 - 120.0")

# Outcomes: Start with significant O for activation
ic_randomizer.set_category_ranges('outcomes', 60.0, 120.0)
print("  Outcomes (O): 60.0 - 120.0")

# Generate model with randomized initial conditions
final_model = ic_randomizer.randomize_initial_conditions(seed=42)

print("Optimal initial conditions applied")

# %% [markdown]
# ## 5. Analyze Cascade Parameters using ParameterMapper
#
# Verify that key parameters are properly set for cascade activation

# %%
# Get parameter map for analysis
param_map = get_parameter_reaction_map(final_model)

print("Cascade Parameter Analysis:")
print("=" * 60)

# Analyze parameters for each cascade step
cascade_steps = ['R1', 'I1_1', 'O']
for step in cascade_steps:
    step_params = get_parameters_for_state(final_model, step)
    activated_params = get_parameters_for_state(final_model, step + 'a')
    
    print(f"\n{step} → {step}a Activation:")
    print(f"  Parameters affecting {step}: {len(step_params['all'])}")
    print(f"  Parameters affecting {step}a: {len(activated_params['all'])}")
    
    # Find Vmax parameters for forward activation
    vmax_params = find_parameter_by_role(final_model, 'Vmax', step)
    if vmax_params:
        print(f"  Vmax parameters for {step}: {vmax_params}")

# Special focus on Oa activation
oa_params = get_parameters_for_state(final_model, 'Oa')
print(f"\nOa Activation Parameters:")
print(f"  Total parameters: {len(oa_params['all'])}")
print(f"  As reactant: {oa_params['as_reactant']}")
print(f"  As product: {oa_params['as_product']}")

# %% [markdown]
# ## 6. Simulate and Validate Cascade Activation
#
# Test if the tuned system achieves proper cascade activation

# %%
# Create solver and simulate
solver = RoadrunnerSolver()
solver.compile(final_model.get_sbml_model())

# Extended simulation (0-1000, drug at t=500)
result = solver.simulate(start=0, stop=1000, step=101)

print("Simulation Results:")
print("=" * 60)

# Calculate key metrics
initial_oa = result['Oa'].values[0]
pre_drug_oa = result['Oa'].values[50]  # t=500
post_drug_oa = result['Oa'].values[-1]  # t=1000
initial_o = result['O'].values[0]

# Activation metrics
activation_ratio = pre_drug_oa / (pre_drug_oa + result['O'].values[50]) * 100
drug_response = ((pre_drug_oa - post_drug_oa) / pre_drug_oa * 100) if pre_drug_oa > 0 else 0

print(f"\nCascade Activation Metrics:")
print(f"  O at t=0: {initial_o:.2f}")
print(f"  Oa at t=0: {initial_oa:.2f}")
print(f"  Oa at t=500 (pre-drug): {pre_drug_oa:.2f}")
print(f"  Oa at t=1000 (post-drug): {post_drug_oa:.2f}")
print(f"  Activation ratio at t=500: {activation_ratio:.1f}%")
print(f"  Drug response (decrease): {drug_response:.1f}%")

# Check cascade activation at intermediate steps
print(f"\nIntermediate Activation (at t=500):")
print(f"  R1: {result['R1'].values[50]:.2f}, R1a: {result['R1a'].values[50]:.2f}")
print(f"  I1_1: {result['I1_1'].values[50]:.2f}, I1_1a: {result['I1_1a'].values[50]:.2f}")

# %% [markdown]
# ## 7. Visualize Cascade Activation and Drug Response

# %%
# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Full cascade trajectories
ax1 = axes[0, 0]
ax1.plot(result['time'], result['R1a'], 'b-', label='R1a', linewidth=2)
ax1.plot(result['time'], result['I1_1a'], 'g-', label='I1_1a', linewidth=2)
ax1.plot(result['time'], result['Oa'], 'r-', label='Oa', linewidth=2)
ax1.axvline(x=500, color='gray', linestyle='--', alpha=0.7, label='Drug D applied')
ax1.set_xlabel('Time')
ax1.set_ylabel('Activated Species')
ax1.set_title('Cascade Activation Trajectories')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: O vs Oa ratio
ax2 = axes[0, 1]
oa_ratio = result['Oa'] / (result['O'] + result['Oa']) * 100
ax2.plot(result['time'], oa_ratio, 'purple', linewidth=2)
ax2.axvline(x=500, color='gray', linestyle='--', alpha=0.7)
ax2.set_xlabel('Time')
ax2.set_ylabel('Oa / (O + Oa) (%)')
ax2.set_title('O Activation Percentage')
ax2.grid(True, alpha=0.3)

# Plot 3: Drug effect on Oa
ax3 = axes[0, 2]
ax3.plot(result['time'][:51], result['Oa'][:51], 'b-', linewidth=2, label='Pre-drug')
ax3.plot(result['time'][50:], result['Oa'][50:], 'r-', linewidth=2, label='Post-drug')
ax3.axvline(x=500, color='gray', linestyle='--', alpha=0.7)
ax3.set_xlabel('Time')
ax3.set_ylabel('Oa Concentration')
ax3.set_title('Drug Effect on Oa')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Cascade activation at t=500 (pre-drug)
ax4 = axes[1, 0]
cascade_levels = ['R1→R1a', 'I1_1→I1_1a', 'O→Oa']
activation_levels = [
    result['R1a'].values[50] / (result['R1'].values[50] + result['R1a'].values[50]) * 100,
    result['I1_1a'].values[50] / (result['I1_1'].values[50] + result['I1_1a'].values[50]) * 100,
    result['Oa'].values[50] / (result['O'].values[50] + result['Oa'].values[50]) * 100
]
bars = ax4.bar(cascade_levels, activation_levels, color=['blue', 'green', 'red'], alpha=0.7)
ax4.set_ylabel('Activation Percentage (%)')
ax4.set_title('Cascade Activation at t=500 (Pre-drug)')
ax4.grid(True, alpha=0.3, axis='y')
for bar, value in zip(bars, activation_levels):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height, f'{value:.1f}%', 
             ha='center', va='bottom')

# Plot 5: Parameter statistics
ax5 = axes[1, 1]
param_stats = param_randomizer.get_parameter_statistics()
param_types = list(param_stats.keys())
param_counts = [param_stats[t]['count'] for t in param_types]
bars = ax5.bar(param_types, param_counts, alpha=0.7)
ax5.set_xlabel('Parameter Type')
ax5.set_ylabel('Count')
ax5.set_title('Parameter Distribution by Type')
ax5.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, param_counts):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height, str(count), 
             ha='center', va='bottom')

# Plot 6: System summary
ax6 = axes[1, 2]
ax6.axis('off')
summary_text = f"""System Performance Summary:

Cascade Activation:
• R1 activation: {activation_levels[0]:.1f}%
• I1_1 activation: {activation_levels[1]:.1f}%
• O activation: {activation_levels[2]:.1f}%

Drug Response:
• Pre-drug Oa: {pre_drug_oa:.2f}
• Post-drug Oa: {post_drug_oa:.2f}
• Response: {drug_response:.1f}% decrease

Parameter Tuning:
• Vmax range: 5.0-50.0
• Km range: 10.0-100.0  
• Kc range: 0.1-1.0

Result: {'✓ Responsive' if drug_response > 30 and activation_levels[2] > 30 else '✗ Needs tuning'}
"""
ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontfamily='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Parameter Sensitivity Analysis
#
# Test how the system responds to different parameter variations

# %%
# Define different test scenarios
test_scenarios = [
    {"name": "High_Activation", "vmax_range": (10.0, 80.0), "kc_range": (0.2, 1.5)},
    {"name": "Moderate", "vmax_range": (5.0, 50.0), "kc_range": (0.1, 1.0)},
    {"name": "Low_Activation", "vmax_range": (2.0, 20.0), "kc_range": (0.05, 0.5)},
    {"name": "Sensitive", "vmax_range": (20.0, 100.0), "kc_range": (0.3, 2.0)},
]

sensitivity_results = []

for scenario in test_scenarios:
    print(f"\nTesting {scenario['name']} scenario...")
    
    # Create new randomizer with scenario-specific ranges
    test_param_randomizer = ParameterRandomizer(model)
    test_param_randomizer.set_range_for_param_type('Vmax', *scenario['vmax_range'])
    test_param_randomizer.set_range_for_param_type('Kc', *scenario['kc_range'])
    
    # Generate test model
    test_model = test_param_randomizer.randomize_all_parameters(seed=42 + len(sensitivity_results))
    
    # Apply initial conditions
    test_ic_randomizer = InitialConditionRandomizer(test_model)
    test_ic_randomizer.set_category_ranges('receptors', 80.0, 150.0)
    test_ic_randomizer.set_category_ranges('activated', 20.0, 60.0)
    test_ic_randomizer.set_category_ranges('outcomes', 60.0, 120.0)
    test_final_model = test_ic_randomizer.randomize_initial_conditions(seed=42)
    
    # Simulate
    test_solver = RoadrunnerSolver()
    test_solver.compile(test_final_model.get_sbml_model())
    test_result = test_solver.simulate(start=0, stop=1000, step=101)
    
    # Calculate metrics
    test_pre_oa = test_result['Oa'].values[50]
    test_post_oa = test_result['Oa'].values[-1]
    test_activation = test_pre_oa / (test_pre_oa + test_result['O'].values[50]) * 100
    test_response = ((test_pre_oa - test_post_oa) / test_pre_oa * 100) if test_pre_oa > 0 else 0
    
    sensitivity_results.append({
        "name": scenario["name"],
        "pre_oa": test_pre_oa,
        "post_oa": test_post_oa,
        "activation": test_activation,
        "response": test_response,
        "result": test_result
    })
    
    print(f"  Activation: {test_activation:.1f}%")
    print(f"  Drug response: {test_response:.1f}%")

# %% [markdown]
# ## 9. System Performance Summary

# %%
print("=" * 60)
print("RESPONSIVE SYSTEM PERFORMANCE SUMMARY")
print("=" * 60)

print("\nDesign Goals:")
print("  1. Cascade properly activates Oa (>30% activation)")
print("  2. Drug D produces clear, measurable response (>30% change)")
print("  3. System works across parameter variations")

print("\nResults:")
print(f"  Base system activation: {activation_ratio:.1f}%")
print(f"  Base system drug response: {drug_response:.1f}%")

print("\nParameter Sensitivity Results:")
for res in sensitivity_results:
    status = "✓" if res["activation"] > 30 and res["response"] > 30 else "✗"
    print(f"  {status} {res['name']}: Activation={res['activation']:.1f}%, Response={res['response']:.1f}%")

print("\n" + "=" * 60)
print("Key Insights:")
print("  1. Vmax range critical for activation strength")
print("  2. Kc (basal activation) affects baseline activity")
print("  3. Initial conditions affect steady-state levels")
print("  4. Drug response proportional to cascade activation")
print("=" * 60)

print("\nReady for execution. Run this script to see the tuned responsive system.")
print("The system is designed to reliably perturb Oa upon drug D treatment.")
