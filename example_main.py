import numpy as np
import matplotlib.pyplot as plt
import helper
from scipy.spatial.distance import squareform, pdist
from ResultVisualizer import ResultVisualizer
from MatchRouteWrapper import MatchRouteWrapper

flag_verbose = True
flag_show_plot = True
folder_name = './temp/'

flag_initialize = 0         # 0: VRP, 1: random
flag_solver = 1             # 0 GUROBI exact solver, 1: OrTool heuristic solver
solver_time_limit = 30.0    # The time limit for the solver
flag_uncertainty = False    # Whether we consider time uncertainty in the optimization. If False, the time costs are assumed deterministic
beta = 0.8                  # A number that will affect the penalty on the time uncertainty, keep it as the default value

# Initialize an small randomized problem
veh_num = 4                 # The number of agents (robots/vehicles/guides)
node_num = 12               # The number of nodes (places of interest + 2). The "2" is due to a start and terminal node
demand_penalty = 1000.0     # The penalty on dropping a human requested POI
time_penalty = 1.0          # The penalty on the total time consumption of the tours
time_limit = 500            # The time limit on the tours
human_num = 10              # The number of human
human_choice = 5            # The maximum number of POIs that a human can request to visit
max_iter = 10               # The maximum number of iterations for the OrTool heuristic solver
max_human_in_team = np.ones(veh_num, dtype=int) * 3 # (human_num // veh_num + 5)  # The maximum number of humans that a robot can guide
place_num = node_num - 2    # The number of places of interest (POI)

# To initialize a larger randomized problem, uncomment the following
# veh_num = 10
# node_num = 50
# demand_penalty = 1000.0
# time_penalty = 1.0
# time_limit = 900
# human_num = 100
# human_choice = 10
# max_iter = 10
# max_human_in_team = np.ones(veh_num, dtype=int) * 15 # (human_num // veh_num + 5)
# place_num = node_num - 2

# Set sequence constraints
# node_seq = None
node_seq = [[0,1,2], [3,4]] # Sequence constraints
                            # example [[0,1,2], [3,4]] means if a human want to visit 1, he/she must visit 0 first,
                            #                                if a human want to visit 2, he/she must visit 1 first;
                            #               apart from that, if a human want to visit 4, he/she must visit 3 first.
                            # Set to None if there are no such constraints

# Initialize the result visualizer
visualizer = ResultVisualizer()

# Initialize spacial maps
node_pose = np.random.rand(node_num, 2) * 200.0 # The size of the world is 200x200
node_pose[-1, :] = 100.0                        # The start location is at (100, 100)
node_pose[-2, :] = 100.0                        # The terminal location is at (100, 100), please keep them the same
edge_dist = squareform(pdist(node_pose))
veh_speed = np.ones(veh_num, dtype=np.float64)
edge_time = edge_dist.reshape(1,node_num,node_num) / veh_speed.reshape(veh_num,1,1)
node_time = np.ones((veh_num,node_num), dtype=np.float64) * 30.0
print('node_pose = ', node_pose)
if flag_uncertainty:
    edge_time_std = edge_time * 0.2
    node_time_std = node_time * 0.2
else:
    edge_time_std = None
    node_time_std = None
print('node_pose = ', node_pose)

# Initialize human selections
# Here we randomly initialize the human requested POIs to visit
global_planner = MatchRouteWrapper(veh_num, node_num, human_choice, human_num, max_human_in_team, demand_penalty, time_penalty, time_limit, solver_time_limit, beta, flag_verbose)
human_demand_bool, human_demand_int_unique = global_planner.initialize_human_demand()
print('human_demand_int_unique = \n', human_demand_int_unique)

# Do optimization
flag_success, route_list, route_time_list, team_list, human_in_team, y_sol, z_sol, result_dict = global_planner.plan(edge_time, node_time, edge_time_std, node_time_std, human_demand_bool, node_seq, max_iter, flag_initialize, flag_solver)
print('sum_obj = demand_penalty * demand_obj + time_penalty * sum_time = %f * %f + %f * %f = %f' % (demand_penalty, result_dict['demand_obj'], time_penalty, result_dict['result_sum_time'], result_dict['sum_obj']))

# Print and visualize some results
human_counts = global_planner.evaluator.count_human(human_in_team, veh_num)
print('human_in_team', human_in_team)
print('human_counts', human_counts)
print('total_demand = ', human_demand_bool.sum())

# See the plots in the folder ./temp/
visualizer.print_results(route_list, route_time_list, team_list)
if flag_show_plot:
    sum_obj_list = result_dict['sum_obj_list']
    demand_obj_list = result_dict['demand_obj_list']
    result_max_time_list = result_dict['result_max_time_list']

    visualizer.visualize_routes(node_pose, route_list)
    visualizer.save_plots(folder_name)
    # visualizer.show_plots()

    iter_range = np.arange(2*max_iter)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(iter_range, sum_obj_list)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective function')
    fig_file = folder_name + "objective.png"
    fig.savefig(fig_file, bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(iter_range, demand_obj_list)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Dropped demand')
    fig_file = folder_name + "demand.png"
    fig.savefig(fig_file, bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(iter_range, result_max_time_list / 10)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max tour time (min)')
    fig_file = folder_name + "maxtime.png"
    fig.savefig(fig_file, bbox_inches='tight')
