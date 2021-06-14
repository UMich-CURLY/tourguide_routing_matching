import numpy as np
from scipy.spatial.distance import squareform, pdist
from GurobiRoutingSolver import GurobiRoutingSolver
from OrtoolRoutingSolver import OrtoolRoutingSolver
from ResultVisualizer import ResultVisualizer

veh_num = 4
node_num = 10
human_num = 10
demand_penalty = 10.0
time_penalty = 1.0
flag_solver_type = 0


# node_pose = np.array( [[ 9.84212669 ,  1.80265101],
#  [13.03963515 , 16.09854408],
#  [19.30844887 , 10.85317018],
#  [14.08816635 ,  8.47962349],
#  [11.54766554 ,  1.88200905],
#  [18.546858   , 14.70094731],
#  [10.         , 10.        ],
#  [10.         , 10.        ]])

node_pose = np.random.rand(node_num, 2) * 20.0
node_pose[-1, :] = 10.0
node_pose[-2, :] = 10.0

print('node_pose = ', node_pose)

edge_dist = squareform(pdist(node_pose))
veh_speed = np.ones(veh_num, dtype=np.float64)

edge_time = edge_dist.reshape(1,node_num,node_num) / veh_speed.reshape(veh_num,1,1)
node_time = np.ones((veh_num,node_num), dtype=np.float64) * 3.0

visualizer = ResultVisualizer()

routing_solver = GurobiRoutingSolver(veh_num, node_num, human_num, demand_penalty, time_penalty, flag_solver_type)
routing_solver.set_model(edge_time, node_time)
routing_solver.set_all_task_complete()
routing_solver.set_objective()
routing_solver.optimize()
route_list, team_list = routing_solver.get_plan()
visualizer.print_results(route_list, team_list)
# visualizer.visualize_routes(node_pose, route_list)


routing_solver = OrtoolRoutingSolver(veh_num, node_num, human_num, demand_penalty, time_penalty, flag_solver_type)
routing_solver.set_model(edge_time, node_time)
routing_solver.optimize()
route_list, team_list = routing_solver.get_plan()

visualizer.print_results(route_list, team_list)
visualizer.visualize_routes(node_pose, route_list)

visualizer.show_plots()

