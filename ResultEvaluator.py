import numpy as np

class ResultEvaluator:
    def __init__(self, veh_num, node_num, human_num, demand_penalty, time_penalty):
        self.veh_num = veh_num
        self.node_num = node_num
        self.human_num = human_num
        self.demand_penalty = demand_penalty
        self.time_penalty = time_penalty
    
    def objective_fcn(self, edge_time, node_time, route_list, team_list):
        result_max_time = 0.0
        node_visit = np.zeros(self.node_num, dtype=int)
        for k in range(self.veh_num):
            if len(route_list) <= 2:
                continue
            route_time = 0.0
            for i in range(len(route_list[k]) - 1):
                node_i = route_list[k][i]
                node_j = route_list[k][i+1]
                route_time += edge_time[k,node_i,node_j] + node_time[k,node_i]
                node_visit[node_i] += 1
            if route_time > result_max_time:
                result_max_time = route_time
        return result_max_time, node_visit





