import numpy as np
from scipy.spatial.distance import squareform, pdist
from RoutingSolver import RoutingSolver
from ResultVisualizer import ResultVisualizer

veh_num = 4
node_num = 8
human_num = 10
demand_penalty = 10.0
time_penalty = 1.0
flag_solver_type = 0


node_pose = np.random.rand(node_num, 2) * 20.0
node_pose[-1, :] = 10.0
node_pose[-2, :] = 10.0

edge_dist = squareform(pdist(node_pose))
veh_speed = np.ones(veh_num, dtype=np.float64)

edge_time = edge_dist.reshape(1,node_num,node_num) / veh_speed.reshape(veh_num,1,1)
node_time = np.ones((veh_num,node_num), dtype=np.float64) * 3.0

routing_solver = RoutingSolver(veh_num, node_num, human_num, demand_penalty, time_penalty, flag_solver_type)
routing_solver.set_gurobi_model(edge_time, node_time)
routing_solver.set_gurobi_all_task_complete()
routing_solver.set_gurobi_objective()
routing_solver.optimize()
route_list = routing_solver.get_gorubi_route()

visualizer = ResultVisualizer()
visualizer.visualize_routes(node_pose, route_list)

