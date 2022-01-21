import numpy as np
from scipy.stats import norm

def norm_VaR(mu, sigma, beta):
    return mu + sigma * norm.ppf(beta)

def norm_CVaR(mu, sigma, beta):
    return mu + sigma / (1.0 - beta) * norm.pdf(norm.ppf(beta))

class ResultEvaluator:
    def __init__(self, veh_num, node_num, human_num, demand_penalty, time_penalty):
        self.veh_num = veh_num
        self.node_num = node_num
        self.human_num = human_num
        self.demand_penalty = demand_penalty
        self.time_penalty = time_penalty
    
    def objective_fcn(self, edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool, flag_dict = False, beta_input = None, edge_time_std = None, node_time_std = None):
        '''
        This function calculate the objective functon value of the whole optimization
        ------------------------------------------------------
        z_sol:             (human_num, veh_num)
        y_sol:             (veh_num, place_num)
        human_demand_bool: (human_num, place_num), i.e. (human_num, node_num - 2)
        '''
        if edge_time_std is None or node_time_std is None:
            beta = None
            # beta = 0.8
            # edge_time_std = edge_time * 0.3
            # node_time_std = node_time * 0.3
        else:
            beta = beta_input

        if (z_sol is None) or (y_sol is None) or (human_demand_bool is None):
            demand_obj = 0.0
        else:
            place_num = self.node_num-2
            penalty_mat = np.zeros((self.veh_num, place_num), dtype=np.float64) # (veh_num, place_num)
            for k in range(self.veh_num):
                for i in range(place_num):
                    penalty_mat[k, i] = (z_sol[:, k] * human_demand_bool[:, i]).sum()
            demand_obj = ((1-y_sol) * penalty_mat).sum()
            # for k in range(self.veh_num):
            #     print(k, np.nonzero(penalty_mat[k]))

        node_visit = np.zeros(self.node_num, dtype=int)
        result_time_list = np.zeros(self.veh_num, dtype=np.float64)
        result_time_cvar = np.zeros(self.veh_num, dtype=np.float64)
        for k in range(self.veh_num):
            if len(route_list[k]) <= 2:
                continue
            route_time = 0.0
            route_var = 0.0
            for i in range(len(route_list[k]) - 1):
                node_i = route_list[k][i]
                node_j = route_list[k][i+1]
                route_time += edge_time[k,node_i,node_j] + node_time[k,node_i]
                node_visit[node_i] += 1
                if beta is not None:
                    route_var += edge_time_std[k,node_i,node_j]**2 + node_time_std[k,node_i]**2
            result_time_list[k] = route_time
            if beta is not None:
                result_time_cvar[k] = norm_CVaR(route_time, np.sqrt(route_var), beta)
                # TODO: This value is correct only when GurobiRoutingSolver.flag_alpha_var == True
                # However, since this result_time_cvar[k] is not used in the paper, for now, this problem is not fixed yet
                # result_time_cvar[k] = norm.cdf((500 - route_time) / np.sqrt(route_var))
        result_sum_time = result_time_list.sum()
        sum_obj = self.demand_penalty * demand_obj + self.time_penalty * result_sum_time
        obj_dict = {}
        obj_dict['result_time_list'] = result_time_list
        obj_dict['result_max_time'] = result_time_list.max()
        obj_dict['result_sum_time'] = result_sum_time
        obj_dict['result_time_cvar'] = result_time_cvar
        obj_dict['result_max_time_cvar'] = result_time_cvar.max()
        obj_dict['result_sum_time_cvar'] = result_time_cvar.sum()
        if flag_dict:
            return sum_obj, demand_obj, result_sum_time, node_visit, obj_dict
        return sum_obj, demand_obj, result_sum_time,  node_visit

    def count_human(self, human_in_team, veh_num):
        '''
        This function calculate the number of humans in each human-robot teams
        ------------------------------------------------------
        Input:
        human_in_team:      int list of size (human_num), each elements indicate which vehicle that human follows
        veh_num:            int, the agent/robot number
        ------------------------------------------------------
        Output:
        human_counts:       int array of size (veh_num,), human_counts[k] stores the number of human in the team of robot k
        '''
        veh_values_temp, human_counts_temp = np.unique(human_in_team, return_counts=True)
        human_counts = np.zeros(veh_num, dtype=int)
        for i_veh in range(human_counts_temp.shape[0]):
            human_counts[ veh_values_temp[i_veh] ] = human_counts_temp[i_veh]
        return human_counts




