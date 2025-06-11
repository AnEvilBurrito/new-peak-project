import re
from collections import OrderedDict

def parse_initial_conditions(text: str) -> OrderedDict:
    """
    Extract initial conditions in the form X(0) = value from the model text.
    Returns an OrderedDict: {variable: value}
    """
    init_vals = OrderedDict()
    pattern = re.compile(r"^([A-Za-z_]\w*)\(0\)\s*=\s*([eE\d\.\+\-]+)")

    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            var, val = match.groups()
            try:
                init_vals[var] = float(val)
            except ValueError:
                continue
    return init_vals

def parse_parameters(text: str) -> OrderedDict:
    """
    Extract parameter assignments of the form name = value.
    Ignores comments and malformed lines.
    Returns an OrderedDict: {parameter: value}
    """
    params = OrderedDict()
    pattern = re.compile(r"^([A-Za-z_]\w*)\s*=\s*([eE\d\.\+\-]+)")

    for line in text.splitlines():
        clean_line = line.split('%')[0].split('#')[0].strip()
        match = pattern.match(clean_line)
        if match:
            key, val = match.groups()
            try:
                params[key] = float(val)
            except ValueError:
                continue
    return params

def parse_input_assignments(text: str) -> list:
    """
    Convert input assignment rules using piecewiseIQM into Antimony's piecewise() format.
    Example:
    IGF = IGF0*piecewiseIQM(1,ge(time,IGF_on),0)
    becomes:
    IGF := piecewise(IGF0, time >= IGF_on, 0)
    """
    assignments = []
    pattern = re.compile(r"^([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\*\s*piecewiseIQM\(\s*1\s*,\s*ge\(time\s*,\s*([A-Za-z_]\w*)\)\s*,\s*0\s*\)")

    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            var, base, trigger = match.groups()
            assignments.append(f"{var} := piecewise({base}, time >= {trigger}, 0)")
    return assignments

def parse_reversible_reactions(text: str) -> list:
    """
    Parse reversible reactions with 'vf' and 'vr' into two Antimony unidirectional reactions.
    Returns a list of reaction strings.
    """
    lines = text.splitlines()
    reactions = []
    i = 0
    reaction_id = 0

    while i < len(lines):
        line = lines[i].strip()

        # Match start of a reaction block
        if "<=>" in line and ":" in line:
            # Extract species and reaction ID
            lhs_rhs, tag = line.split(":")
            lhs, rhs = lhs_rhs.split("<=>")
            lhs = lhs.strip()
            rhs = rhs.strip()
            tag = tag.strip()

            # Look ahead for vf and vr
            vf_line = lines[i+1].strip()
            vr_line = lines[i+2].strip()

            vf_expr = vf_line.split("=", 1)[1].strip()
            vr_expr = vr_line.split("=", 1)[1].strip()

            reactions.append(f"J{reaction_id}_f: {lhs} -> {rhs}; {vf_expr}")
            reactions.append(f"J{reaction_id}_r: {rhs} -> {lhs}; {vr_expr}")

            i += 3  # skip vf and vr lines
            reaction_id += 1
        else:
            i += 1

    return reactions

def resolve_piecewise_constants(assignments: list, parameters: dict) -> list:
    """
    Replace parameter names in piecewise statements with their actual values.
    Assumes format like:
        X := piecewise(BASE, time >= TRIGGER, 0)
    """
    resolved = []
    pattern = re.compile(r"^([A-Za-z_]\w*) := piecewise\(([^,]+), time >= ([^,]+), 0\)$")

    for line in assignments:
        match = pattern.match(line.strip())
        if match:
            var, base, trigger = match.groups()
            base_val = parameters.get(base, base)
            trigger_val = parameters.get(trigger, trigger)

            resolved_line = f"{var} := piecewise({base_val}, time >= {trigger_val}, 0)"
            resolved.append(resolved_line)
        else:
            resolved.append(line)  # fallback if no match
    return resolved


def assemble_antimony_model(model_text: str) -> str:
    """
    Assemble full Antimony model from model text, including:
    - reactions (as unidirectional pairs)
    - initial conditions
    - parameters
    - input assignments (with inlined constants)
    """
    # Step-by-step parser functions
    init_conds = parse_initial_conditions(model_text)
    parameters = parse_parameters(model_text)
    assignments = parse_input_assignments(model_text)
    reactions = parse_reversible_reactions(model_text)

    # Resolve constants in piecewise assignments
    resolved_assignments = resolve_piecewise_constants(assignments, parameters)

    lines = []
    lines.append("model FGFR4_RTK_model\n")

    # Reactions
    lines.append("# Reactions")
    lines.extend(reactions)
    lines.append("")

    # Initial conditions
    lines.append("# Initial Conditions")
    for var, val in init_conds.items():
        lines.append(f"{var} = {val}")
    lines.append("")

    # Parameters
    lines.append("# Parameters")
    for key, val in parameters.items():
        lines.append(f"{key} = {val}")
    lines.append("")

    # Assignment rules (resolved)
    lines.append("# Input Functions")
    lines.extend(resolved_assignments)
    lines.append("")

    lines.append("end")
    return "\n".join(lines)


