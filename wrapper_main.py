import numpy as np
import matplotlib.pyplot as plt
import helper
from scipy.spatial.distance import squareform, pdist
from ResultVisualizer import ResultVisualizer
from MatchRouteWrapper import MatchRouteWrapper

flag_verbose = True
flag_show_plot = True
folder_name = './temp/'

flag_read_testcase = True
flag_save_testcase = False
testcase_file = './testcase/case.dat'
flag_initialize = 0 # 0: VRP, 1: random

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

veh_num = 4
node_num = 12
demand_penalty = 1000.0
time_penalty = 1.0
time_limit = 500
human_num = 10
human_choice = 5
max_iter = 10
max_human_in_team = np.ones(veh_num, dtype=int) * 3 # (human_num // veh_num + 5)
place_num = node_num - 2

# node_seq = None
node_seq = [[0,1,2], [3,4]]

global_planner = MatchRouteWrapper(veh_num, node_num, human_choice, human_num, max_human_in_team, demand_penalty, time_penalty, time_limit, flag_verbose)

if flag_read_testcase:
    data_dict = helper.load_dict(testcase_file)
    node_pose = data_dict['node_pose']
    human_demand_int = data_dict['human_demand_int']
else:
    node_pose = np.random.rand(node_num, 2) * 200.0
    human_demand_int = np.random.randint(0, place_num, (human_num,human_choice) )
if flag_save_testcase:
    data_dict = {'node_pose': node_pose, 'human_demand_int': human_demand_int}
    helper.save_dict(testcase_file, data_dict)

# Initialize spacial maps
node_pose[-1, :] = 100.0
node_pose[-2, :] = 100.0

edge_dist = squareform(pdist(node_pose))
veh_speed = np.ones(veh_num, dtype=np.float64)

edge_time = edge_dist.reshape(1,node_num,node_num) / veh_speed.reshape(veh_num,1,1)
node_time = np.ones((veh_num,node_num), dtype=np.float64) * 30.0
print('node_pose = ', node_pose)

# Initialize human selections
human_demand_bool, human_demand_int_unique = global_planner.initialize_human_demand(human_demand_int)
print('human_demand_int_unique = \n', human_demand_int_unique)

# Initialize the visualizer and evaluator
visualizer = ResultVisualizer()

# Initialize an routing plan
route_list, route_time_list, team_list, y_sol = global_planner.initialize_plan(edge_time, node_time, flag_initialize)

visualizer.print_results(route_list, route_time_list, team_list)
if flag_show_plot:
    visualizer.visualize_routes(node_pose, route_list)

flag_success, route_list, route_time_list, team_list, human_in_team, y_sol, z_sol, sum_obj_list, demand_obj_list, result_max_time_list = global_planner.generate_plan(edge_time, node_time, human_demand_bool, route_list, y_sol, node_seq, max_iter)

human_counts = global_planner.evaluator.count_human(human_in_team, veh_num)
print('human_in_team', human_in_team)
print('human_counts', human_counts)
print('total_demand = ', human_demand_bool.sum())

visualizer.print_results(route_list, route_time_list, team_list)
if flag_show_plot:
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