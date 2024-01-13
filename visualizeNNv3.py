import pickle
import matplotlib.pyplot as plt
import neat
import numpy as np

def plot_neuron(ax, center, theta, radius=0.5, color='blue'):
    circle = plt.Circle(center, radius, color=color, fill=False, lw=1)
    ax.add_patch(circle)

    # Extending lines
    x, y = center
    for angle in theta:
        ax.plot([x, x + radius * np.cos(angle)], [y, y + radius * np.sin(angle)], color='black', lw=1)

def visualize_genome(config, genome):
    # Create network from genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fig, ax = plt.subplots()
    nodes = {}
    max_layers = max(node.layer for node in net.node_evals) + 1
    total_height = 10.0
    radius = 5.0 / max_layers
    theta = np.linspace(0, 2 * np.pi, 10)

    for node in net.node_evals:
        layer = node.layer
        x_offset = 10.0 * layer
        if layer not in nodes:
            nodes[layer] = []
        nodes[layer].append(node)

    for layer, layer_nodes in nodes.items():
        vertical_spacing = total_height / (len(layer_nodes) + 1)
        for i, node in enumerate(layer_nodes):
            center = (x_offset, vertical_spacing * (i + 1))
            plot_neuron(ax, center, theta, radius, color='green' if node.response < 0 else 'blue')

            for conn in node.connections:
                input_node = conn[0]
                input_layer = net.node_evals[input_node].layer
                input_x_offset = 10.0 * input_layer
                input_index = nodes[input_layer].index(net.node_evals[input_node])
                input_center_y = vertical_spacing * (input_index + 1)

                color = 'red' if conn[2] < 0 else 'green'
                ax.plot([input_x_offset + radius, center[0] - radius], [input_center_y, center[1]], color=color, lw=1)

    plt.axis('equal')
    plt.axis('off')
    plt.show()

# Load configuration and genome
config_path = 'config-feedforward-recLogic'
with open('best_genome_recLogic-BestModel.pkl', 'rb') as f:
    genome = pickle.load(f)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

visualize_genome(config, genome)
