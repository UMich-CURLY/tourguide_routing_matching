import numpy as np
import matplotlib.pyplot as plt
import helper
from scipy.spatial.distance import squareform, pdist
from GurobiRoutingSolver import GurobiRoutingSolver
from OrtoolRoutingSolver import OrtoolRoutingSolver
from ResultVisualizer import ResultVisualizer
from ResultEvaluator import ResultEvaluator
from OrtoolHumanMatcher import OrtoolHumanMatcher

flag_verbose = False
flag_show_plot = True
folder_name = './temp/'

flag_read_testcase = False
flag_save_testcase = True
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
temp_index = np.arange(human_num).reshape(-1,1).repeat(human_choice,axis=1)
human_demand_bool = np.zeros((human_num, place_num), dtype=np.float64)
human_demand_bool[temp_index.reshape(-1), human_demand_int.reshape(-1)] = 1.0
human_demand_int_unique = []
for l in range(human_num):
    human_demand_int_unique.append(np.unique(human_demand_int[l]))
print('human_demand_int_unique = \n', human_demand_int_unique)

# Initialize the visualizer and evaluator
visualizer = ResultVisualizer()
evaluator = ResultEvaluator(veh_num, node_num, human_num, demand_penalty, time_penalty)

# Initialize an routing plan
routing_solver = OrtoolRoutingSolver(veh_num, node_num, human_num, demand_penalty, time_penalty, time_limit)
if flag_initialize == 0:
    routing_solver.set_model(edge_time, node_time)
    routing_solver.optimize()
    route_list, route_time_list, team_list, y_sol = routing_solver.get_plan()
else:
    # routing_solver.set_model(edge_time, node_time)
    # routing_solver.optimize()
    # route_list, route_time_list, team_list, y_sol = routing_solver.get_plan()
    route_list, route_time_list, team_list, y_sol = routing_solver.get_random_plan(edge_time, node_time)

z_sol = None
visualizer.print_results(route_list, route_time_list, team_list)
if flag_show_plot:
    visualizer.visualize_routes(node_pose, route_list)
sum_obj, demand_obj, result_max_time, node_visit = evaluator.objective_fcn(edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool)
print('sum_obj = demand_penalty * demand_obj + time_penalty * max_time = %f * %f + %f * %f = %f' % (demand_penalty, demand_obj, time_penalty, result_max_time, sum_obj))
# print('node_visit = ', node_visit)

sum_obj_list = np.empty(2*max_iter, dtype=np.float64)
demand_obj_list = np.empty(2*max_iter, dtype=np.float64)
result_max_time_list = np.empty(2*max_iter, dtype=np.float64)
for i_iter in range(max_iter):
    human_matcher = OrtoolHumanMatcher(human_num, veh_num, max_human_in_team)
    human_in_team, z_sol, demand_result = human_matcher.optimize(human_demand_bool, y_sol)
    sum_obj, demand_obj, result_max_time, node_visit = evaluator.objective_fcn(edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool)
    # print('human_in_team', human_in_team)
    # print('z_sol', z_sol)
    # print('demand_result = ', demand_result)
    print('sum_obj1 = demand_penalty * demand_obj + time_penalty * max_time = %f * %f + %f * %f = %f ... (1)' % (demand_penalty, demand_obj, time_penalty, result_max_time, sum_obj))
    # print('node_visit = ', node_visit)
    sum_obj_list[2*i_iter] = sum_obj
    demand_obj_list[2*i_iter] = demand_obj
    result_max_time_list[2*i_iter] = result_max_time

    if (flag_initialize != 0) and (i_iter == 0):
        route_list = None
    result_dict = routing_solver.optimize_sub(edge_time, node_time, z_sol, human_demand_bool, node_seq, route_list)
    route_list, route_time_list, team_list, y_sol = routing_solver.get_plan(flag_sub_solver=True)
    sum_obj, demand_obj, result_max_time, node_visit = evaluator.objective_fcn(edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool)
    print('sum_obj2 = demand_penalty * demand_obj + time_penalty * max_time = %f * %f + %f * %f = %f' % (demand_penalty, demand_obj, time_penalty, result_max_time, sum_obj))
    # print('node_visit = ', node_visit)
    sum_obj_list[2*i_iter+1] = sum_obj
    demand_obj_list[2*i_iter+1] = demand_obj
    result_max_time_list[2*i_iter+1] = result_max_time


human_counts = evaluator.count_human(human_in_team, veh_num)
print('human_in_team', human_in_team)
print('human_counts', human_counts)
print('totam_demand = ', human_demand_bool.sum())

visualizer.print_results(route_list, route_time_list, team_list)
if flag_show_plot:
    visualizer.visualize_routes(node_pose, route_list)
    # visualizer.show_plots()
    visualizer.save_plots(folder_name)

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
