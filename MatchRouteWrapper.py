import numpy as np
from OrtoolRoutingSolver import OrtoolRoutingSolver
from OrtoolHumanMatcher import OrtoolHumanMatcher
from ResultEvaluator import ResultEvaluator

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
        self.flag_verbose = True

        self.flag_initialize = 0

        self.routing_solver = OrtoolRoutingSolver(veh_num, node_num, human_num, demand_penalty, time_penalty, time_limit)
        self.human_matcher = OrtoolHumanMatcher(human_num, veh_num, max_human_in_team)
        self.evaluator = ResultEvaluator(veh_num, node_num, human_num, demand_penalty, time_penalty)

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
        '''
        flag_initialize: 0 means initialize the routes for the vehicles by solving a routing problem
                         1 means initialize the routes randomly
        '''
        # Initialize an routing plan
        self.flag_initialize = flag_initialize
        if flag_initialize == 0:
            self.routing_solver.set_model(edge_time, node_time)
            self.routing_solver.optimize()
            route_list, route_time_list, team_list, y_sol = self.routing_solver.get_plan()
        else:
            route_list, route_time_list, team_list, y_sol = self.routing_solver.get_random_plan(edge_time, node_time)
        return route_list, route_time_list, team_list, y_sol


    def generate_plan(self, edge_time, node_time, human_demand_bool, route_list_initial, y_sol_inital, node_seq, max_iter):
        y_sol = y_sol_inital + 0
        route_list = route_list_initial # Shallow copy
        sum_obj_list = np.empty(2*max_iter, dtype=np.float64)
        demand_obj_list = np.empty(2*max_iter, dtype=np.float64)
        result_max_time_list = np.empty(2*max_iter, dtype=np.float64)

        z_sol = None
        if self.flag_verbose:
            sum_obj, demand_obj, result_max_time, node_visit = self.evaluator.objective_fcn(edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool)
            print('sum_obj = demand_penalty * demand_obj + time_penalty * max_time = %f * %f + %f * %f = %f' % (self.demand_penalty, demand_obj, self.time_penalty, result_max_time, sum_obj))
        for i_iter in range(max_iter):
            # self.human_matcher = OrtoolHumanMatcher(self.human_num, self.veh_num, self.max_human_in_team)
            temp_flag_success, human_in_team, z_sol, demand_result = self.human_matcher.optimize(human_demand_bool, y_sol)
            if not temp_flag_success:
                break
            sum_obj, demand_obj, result_max_time, node_visit = self.evaluator.objective_fcn(edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool)
            if self.flag_verbose:
                print('sum_obj1 = demand_penalty * demand_obj + time_penalty * max_time = %f * %f + %f * %f = %f ... (1)' % (self.demand_penalty, demand_obj, self.time_penalty, result_max_time, sum_obj))
            sum_obj_list[2*i_iter] = sum_obj
            demand_obj_list[2*i_iter] = demand_obj
            result_max_time_list[2*i_iter] = result_max_time

            if (self.flag_initialize != 0) and (i_iter == 0):
                # if i_iter == 0:
                route_list = None
            temp_flag_success, result_dict = self.routing_solver.optimize_sub(edge_time, node_time, z_sol, human_demand_bool, node_seq, route_list)
            if not temp_flag_success:
                break
            route_list, route_time_list, team_list, y_sol = self.routing_solver.get_plan(flag_sub_solver=True)
            sum_obj, demand_obj, result_max_time, node_visit = self.evaluator.objective_fcn(edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool)
            if self.flag_verbose:
                print('sum_obj2 = demand_penalty * demand_obj + time_penalty * max_time = %f * %f + %f * %f = %f' % (self.demand_penalty, demand_obj, self.time_penalty, result_max_time, sum_obj))
            sum_obj_list[2*i_iter+1] = sum_obj
            demand_obj_list[2*i_iter+1] = demand_obj
            result_max_time_list[2*i_iter+1] = result_max_time
        flag_success = i_iter >= 1
        return flag_success, route_list, route_time_list, team_list, human_in_team, y_sol, z_sol, sum_obj_list, demand_obj_list, result_max_time_list
