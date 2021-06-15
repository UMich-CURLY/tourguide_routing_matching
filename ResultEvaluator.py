import numpy as np

class ResultEvaluator:
    def __init__(self, veh_num, node_num, human_num, demand_penalty, time_penalty):
        self.veh_num = veh_num
        self.node_num = node_num
        self.human_num = human_num
        self.demand_penalty = demand_penalty
        self.time_penalty = time_penalty
    
    def objective_fcn(self, edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool):
        '''
        z_sol:             (human_num, veh_num)
        y_sol:             (veh_num, place_num)
        human_demand_bool: (human_num, place_num), i.e. (human_num, node_num - 2)
        '''
        if (z_sol is None) or (y_sol is None) or (human_demand_bool is None):
            demand_obj = 0.0
        else:
            place_num = self.node_num-2
            penalty_mat = np.zeros((self.veh_num, place_num), dtype=np.float64) # (veh_num, place_num)
            for k in range(self.veh_num):
                for i in range(place_num):
                    penalty_mat[k, i] = (z_sol[:, k] * human_demand_bool[:, i]).sum()
            demand_obj = ((1-y_sol) * penalty_mat).sum()

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
        sum_obj = self.demand_penalty * demand_obj + self.time_penalty * result_max_time
        return sum_obj, demand_obj, result_max_time,  node_visit





