import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass

def create_layers(num_intermediate_layers, num_cascades, drugs):
    '''
    Create layers based on the number of cascades and regulations.
    '''
    layers = {}
    layers['Drugs'] = drugs
    layers['Receptors'] = [f'R{i+1}' for i in range(num_cascades)]
    for i in range(num_intermediate_layers):
        layers[f'Layer{i+1}'] = [f'I{i+1}_{j+1}' for j in range(num_cascades)]
    layers['Outcome'] = ['O']
    return layers

def calculate_positions(layers):
    """Dynamically calculate node positions based on layers structure"""
    pos = {}
    vertical_spacing = 1.0 / len(layers)
    layer_y_positions = [1.0 - i*vertical_spacing for i in range(len(layers))]
    max_horizontal_nodes = max(len(nodes) for nodes in layers.values())
    
    for i, (layer_name, nodes) in enumerate(layers.items()):
        y = layer_y_positions[i]
        horizontal_spacing = 1.5 / max(len(nodes), 1)
        
        # Center-align nodes per layer
        x_start = -(len(nodes)-1)*horizontal_spacing/2
        for j, node in enumerate(nodes):
            pos[node] = (x_start + j*horizontal_spacing, y)
            
    return pos

def visualise_network(layers, regulations, feedback_regulations=None):
    """Visualise network with custom edge coloring feedback"""
    # Create graph and positions following your layer logic [1][4]
    G = nx.DiGraph()
    all_regulations = regulations + (feedback_regulations if feedback_regulations else [])
    
    # Populate graph with all regulations
    for regulation in all_regulations:
        G.add_node(regulation.from_specie)
        G.add_node(regulation.to_specie)
        G.add_edge(regulation.from_specie, regulation.to_specie, 
                   regulation_type=regulation.reg_type)
    
    # Create quick lookup structure for feedback regulations
    feedback_edges = set()
    if feedback_regulations:
        feedback_edges = {(r.from_specie, r.to_specie) for r in feedback_regulations}
    
    # Position calculation per your dynamic system
    pos = calculate_positions(layers)
    
    # Dynamic figure sizing based on layers [1]
    fig_width, fig_height = 8 + 1 * len(layers), 6 + 1 * len(layers)
    plt.figure(figsize=(fig_width, fig_height))
    
    # Node drawing (unchanged)
    nx.draw_networkx_nodes(G, pos, node_color='white', 
                           edgecolors='black', node_size=2000, linewidths=1.5)
    
    # EDGE DRAWING WITH NEW COLOR LOGIC
    for edge in G.edges(data=True):
        src, dst, data = edge
        is_feedback = (src, dst) in feedback_edges
        
        # Determine styling parameters
        if is_feedback:
            edge_color = 'blue' if data['regulation_type'] == 'up' else 'red'
        else:  # Non-feedback edge
            edge_color = 'black'
        
        arrowstyle = '-|>' if data['regulation_type'] == 'up' else '-['
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(src, dst)],
            edge_color=edge_color,
            width=2,
            arrowstyle=arrowstyle,
            arrowsize=20 if data['regulation_type'] == 'up' else 10,
            min_target_margin=24,
        )
    
    # Labels and finishing
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    plt.title("Regulatory Network with Feedback Highlighting", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

def calculate_positions_2(layers):
    """Dynamically calculate positions with even spacing"""
    pos = {}
    vertical_spacing = 1.0 / len(layers)
    layer_y_positions = [1.0 - i*vertical_spacing for i in range(len(layers))]
    
    for i, (layer_name, nodes) in enumerate(layers.items()):
        y = layer_y_positions[i]
        num_nodes = len(nodes)
        horizontal_spacing = 1.5 / max(num_nodes, 1)
        
        # Center-align nodes in layer
        x_start = - (num_nodes - 1) * horizontal_spacing / 2
        for j, node in enumerate(nodes):
            pos[node] = (x_start + j*horizontal_spacing, y)
            
    return pos

def visualise_network_2(layers, regulations, feedback_regulations=None):
    """Final visualization with smart edge handling"""
    # Create graph and positions
    G = nx.DiGraph()
    all_regulations = regulations + (feedback_regulations or [])
    
    # Populate graph
    for reg in all_regulations:
        G.add_node(reg.from_specie)
        G.add_node(reg.to_specie)
        G.add_edge(reg.from_specie, reg.to_specie, 
                  regulation_type=reg.reg_type)

    # Identify special edges
    bidirectional = set()
    cross_layer = set()
    same_layer = set()

    # Get layer mappings
    node_to_layer = {}
    for layer_name, nodes in layers.items():
        for node in nodes:
            node_to_layer[node] = layer_name

    for u, v in G.edges():
        # Bidirectional check
        if G.has_edge(v, u):
            bidirectional.add((u, v))
            
        # Layer relationship check
        u_layer = list(layers.keys()).index(node_to_layer[u])
        v_layer = list(layers.keys()).index(node_to_layer[v])
        
        if u_layer == v_layer:
            same_layer.add((u, v))
        elif abs(u_layer - v_layer) > 1:
            cross_layer.add((u, v))

    # Position calculation
    pos = calculate_positions_2(layers)
    feedback_edges = {(r.from_specie, r.to_specie) for r in (feedback_regulations or [])}

    # Dynamic figure sizing
    num_layers = len(layers)
    max_nodes = max(len(nodes) for nodes in layers.values())
    plt.figure(figsize=(8 + max_nodes*1.2, 4 + num_layers*1.5))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='white', 
                          edgecolors='black', node_size=2500, linewidths=1.5)

    # Draw edges with smart curvature
    for edge in G.edges(data=True):
        u, v, data = edge
        is_feedback = (u, v) in feedback_edges
        is_bidi = (u, v) in bidirectional
        is_cross = (u, v) in cross_layer
        is_same_layer = (u, v) in same_layer
        
        # Determine styling parameters
        if is_feedback:
            edge_color = 'blue' if data['regulation_type'] == 'up' else 'red'
        else:
            edge_color = 'black'
            
        arrowstyle = '-|>' if data['regulation_type'] == 'up' else '-['
        
        # Calculate curvature parameters
        if is_bidi:
            rad = 0.25  # Alternate direction for bidirectional
        elif is_cross:
            rad = 0.3
        elif is_same_layer:
            rad = 0.4
        else:
            rad = 0.0
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=edge_color,
            width=2.5 if is_feedback else 1.8,
            style='solid',
            arrowstyle=arrowstyle,
            arrowsize=25 if data['regulation_type'] == 'up' else 10,
            node_size=2500,
            min_target_margin=15,
            connectionstyle=f"arc3,rad={str(rad)}"
        )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, 
                           font_weight='bold', 
                           verticalalignment='center')

    plt.title("Regulatory Network Visualization", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

## Usage example with your parameters
# layers = create_layers(num_intermediate_layers=2, num_cascades=3, drugs=['D'])
# regulations = new_spec.get_regulations()  
# feedback_regulations = new_spec.get_feedback_regulations()  
# visualize_network(layers, regulations, feedback_regulations=feedback_regulations)