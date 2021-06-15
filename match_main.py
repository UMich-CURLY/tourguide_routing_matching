import numpy as np
from scipy.spatial.distance import squareform, pdist
from GurobiRoutingSolver import GurobiRoutingSolver
from OrtoolRoutingSolver import OrtoolRoutingSolver
from ResultVisualizer import ResultVisualizer
from ResultEvaluator import ResultEvaluator
from OrtoolHumanMatcher import OrtoolHumanMatcher

veh_num = 4
node_num = 10
demand_penalty = 10.0
time_penalty = 1.0
flag_solver_type = 0

human_num = 10
human_choice = 5

max_human_in_team = np.ones(veh_num, dtype=int) * (human_num // veh_num + 1)
place_num = node_num - 2

# Initialize spacial maps
node_pose = np.random.rand(node_num, 2) * 200.0
node_pose[-1, :] = 100.0
node_pose[-2, :] = 100.0

edge_dist = squareform(pdist(node_pose))
veh_speed = np.ones(veh_num, dtype=np.float64)

edge_time = edge_dist.reshape(1,node_num,node_num) / veh_speed.reshape(veh_num,1,1)
node_time = np.ones((veh_num,node_num), dtype=np.float64) * 30.0
print('node_pose = ', node_pose)

# Initialize human selections
human_demand_int = np.random.randint(0, place_num, (human_num,human_choice) )
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


routing_solver = OrtoolRoutingSolver(veh_num, node_num, human_num, demand_penalty, time_penalty, flag_solver_type)
routing_solver.set_model(edge_time, node_time)
routing_solver.optimize()
route_list, route_time_list, team_list, y_sol = routing_solver.get_plan()

visualizer.print_results(route_list, route_time_list, team_list)
visualizer.visualize_routes(node_pose, route_list)
result_max_time, node_visit = evaluator.objective_fcn(edge_time, node_time, route_list, team_list)
print('result_max_time = ', result_max_time)
print('node_visit = ', node_visit)

human_matcher = OrtoolHumanMatcher(human_num, veh_num, max_human_in_team)
human_in_team, z_sol, demand_result = human_matcher.optimize(human_demand_bool, y_sol)
print('human_in_team', human_in_team)
print('z_sol', z_sol)
print('demand_result = ', demand_result)


result_dict = routing_solver.set_sub_model(edge_time, node_time, z_sol, human_demand_bool)



# visualizer.show_plots()

