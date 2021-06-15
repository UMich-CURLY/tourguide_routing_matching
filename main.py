import numpy as np
from scipy.spatial.distance import squareform, pdist
from GurobiRoutingSolver import GurobiRoutingSolver
from OrtoolRoutingSolver import OrtoolRoutingSolver
from ResultVisualizer import ResultVisualizer
from ResultEvaluator import ResultEvaluator

veh_num = 4
node_num = 10
human_num = 10
demand_penalty = 10.0
time_penalty = 1.0
time_limit = 0


# node_pose = np.array( [[ 9.84212669 ,  1.80265101],
#  [13.03963515 , 16.09854408],
#  [19.30844887 , 10.85317018],
#  [14.08816635 ,  8.47962349],
#  [11.54766554 ,  1.88200905],
#  [18.546858   , 14.70094731],
#  [10.         , 10.        ],
#  [10.         , 10.        ]])

# node_pose = np.array( [[ 17.87230637 , 35.1245531 ],
#  [ 84.76256192 ,198.77716023],
#  [151.94962027 , 78.23502452],
#  [ 25.62683298 ,154.40066714],
#  [ 43.72012928 ,161.59792387],
#  [189.77192707 ,169.84716883],
#  [194.84484267 ,173.51820618],
#  [160.98953757 , 58.55608649],
#  [100.         ,100.        ],
#  [100.         ,100.        ]])

node_pose = np.random.rand(node_num, 2) * 200.0
node_pose[-1, :] = 100.0
node_pose[-2, :] = 100.0

print('node_pose = ', node_pose)

edge_dist = squareform(pdist(node_pose))
veh_speed = np.ones(veh_num, dtype=np.float64)

edge_time = edge_dist.reshape(1,node_num,node_num) / veh_speed.reshape(veh_num,1,1)
node_time = np.ones((veh_num,node_num), dtype=np.float64) * 30.0

# print(edge_time, node_time)

visualizer = ResultVisualizer()
evaluator = ResultEvaluator(veh_num, node_num, human_num, demand_penalty, time_penalty)

routing_solver = GurobiRoutingSolver(veh_num, node_num, human_num, demand_penalty, time_penalty, time_limit)
routing_solver.set_model(edge_time, node_time)
routing_solver.set_all_task_complete()
routing_solver.set_objective()
routing_solver.optimize()
route_list, route_time_list, team_list, y_sol = routing_solver.get_plan()
visualizer.print_results(route_list, route_time_list, team_list)
visualizer.visualize_routes(node_pose, route_list)
sum_obj, demand_obj, result_max_time, node_visit = evaluator.objective_fcn(edge_time, node_time, route_list, None, y_sol, None)
print('result_max_time = ', result_max_time)
print('node_visit = ', node_visit)

routing_solver = OrtoolRoutingSolver(veh_num, node_num, human_num, demand_penalty, time_penalty, time_limit)
routing_solver.set_model(edge_time, node_time)
routing_solver.optimize()
route_list, route_time_list, team_list, y_sol = routing_solver.get_plan()

visualizer.print_results(route_list, route_time_list, team_list)
visualizer.visualize_routes(node_pose, route_list)
sum_obj, demand_obj, result_max_time, node_visit = evaluator.objective_fcn(edge_time, node_time, route_list, None, y_sol, None)
print('result_max_time = ', result_max_time)
print('node_visit = ', node_visit)

visualizer.show_plots()

