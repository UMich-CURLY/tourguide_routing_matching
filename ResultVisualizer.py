import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import amax, argpartition

class ResultVisualizer:
    def __init__(self):
        self.the_color = [[0,0.447000000000000,0.741000000000000],
                        [0.850000000000000,0.325000000000000,0.0980000000000000],
                        [0.929000000000000,0.694000000000000,0.125000000000000],
                        [0.494000000000000,0.184000000000000,0.556000000000000],
                        [0.466000000000000,0.674000000000000,0.188000000000000],
                        [0.301000000000000,0.745000000000000,0.933000000000000],
                        [0.635000000000000,0.0780000000000000,0.184000000000000]]
    
    def visualize_routes(self, node_pose, route_list):
        veh_num = len(route_list)
        node_num = node_pose.shape[0]
        graph_list = []
        for k in range(veh_num):
            route_len = len(route_list[k])
            if route_len <= 1:
                graph_list.append(None)
                continue
            adjacent_mat = np.zeros((node_num, node_num), dtype=int)
            for i in range(route_len-1):
                node_i = route_list[k][i]
                node_j = route_list[k][i+1]
                adjacent_mat[node_i, node_j] = 1
            a_graph = nx.DiGraph(adjacent_mat)
            graph_list.append(a_graph)
        fixed_positions = {}
        node_colors = []
        node_labels = {}
        for i in range(node_num):
            fixed_positions[i] = node_pose[i, :]
            if i < node_num - 2:
                node_colors.append((0,0,0,0))
                node_labels[i] = i
            elif i <= node_num -2:
                node_colors.append('green')
                node_labels[i] = 'D'
            else:
                node_colors.append('green')
                node_labels[i] = ''
        # fixed_nodes = fixed_positions.keys()
        # pos = nx.spring_layout(G,pos=fixed_positions, fixed = fixed_nodes)
        flag_graph_initialized = False
        plt.figure()
        for k in range(veh_num):
            a_graph = graph_list[k]
            if a_graph is None:
                continue
            fixed_positions1 = {}
            for i in range(node_num):
                fixed_positions1[i] = node_pose[i, :] + 0.1 * (k - veh_num/2)
            arcs = nx.draw_networkx_edges(a_graph, pos=fixed_positions1, edge_color=self.the_color[k%len(self.the_color)])
            if not flag_graph_initialized:
                flag_graph_initialized = True
                nx.draw_networkx_nodes(a_graph, pos=fixed_positions, node_color=node_colors, edgecolors='k')
                nx.draw_networkx_labels(a_graph, pos=fixed_positions, labels=node_labels)
        # nx.draw_networkx(G, node_color=colours)
        plt.show()
        


