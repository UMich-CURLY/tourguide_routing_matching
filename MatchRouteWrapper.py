import numpy as np
from OrtoolRoutingSolver import OrtoolRoutingSolver
from OrtoolHumanMatcher import OrtoolHumanMatcher

class MatchRouteWrapper:
    def __init__(self, veh_num, node_num, human_choice, human_num, max_human_in_team, demand_penalty, time_penalty, time_limit):
        '''
        veh_num: int, the agent number
        node_num: int, the number of nodes, it is assumed that the depot 
        human_choice: int, the maximum number of place
        human_num: int
        max_human_in_team: int -> (veh_num, )
        demand_penalty: float
        time_penalty: float
        time_limit: float
        '''
        self.veh_num = veh_num
        self.node_num = node_num
        self.place_num = node_num - 2
        self.human_choice = human_choice
        self.human_num = human_num
        self.max_human_in_team = max_human_in_team + 0
        self.demand_penalty = demand_penalty
        self.time_penalty = time_penalty
        self.time_limit = time_limit

        self.routing_solver = OrtoolRoutingSolver(self.veh_num, self.node_num, self.human_num, self.demand_penalty, self.time_penalty, self.time_limit)

    def initialize_human_demand(self, human_demand_int = None):
        if human_demand_int is None:
            human_demand_int = np.random.randint(0, self.place_num, (self.human_num,self.human_choice) )
        temp_index = np.arange(self.human_num).reshape(-1,1).repeat(self.human_choice,axis=1)
        human_demand_bool = np.zeros((self.human_num, self.place_num), dtype=np.float64)
        human_demand_bool[temp_index.reshape(-1), human_demand_int.reshape(-1)] = 1.0
        human_demand_int_unique = []
        for l in range(self.human_num):
            human_demand_int_unique.append(np.unique(human_demand_int[l]))
        return human_demand_bool, human_demand_int_unique

    def initialize_plan(self, edge_time, node_time, flag_initialize = 0):
        # Initialize an routing plan
        if flag_initialize == 0:
            self.routing_solver.set_model(edge_time, node_time)
            self.routing_solver.optimize()
            route_list, route_time_list, team_list, y_sol = self.routing_solver.get_plan()
        else:
            route_list, route_time_list, team_list, y_sol = self.routing_solver.get_random_plan(edge_time, node_time)
        return route_list, route_time_list, team_list, y_sol
