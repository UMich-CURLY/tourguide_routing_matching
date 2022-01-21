import time
import numpy as np
from OrtoolRoutingSolver import OrtoolRoutingSolver
from OrtoolHumanMatcher import OrtoolHumanMatcher
from ResultEvaluator import ResultEvaluator
from GurobiRoutingSolver import GurobiRoutingSolver

class MatchRouteWrapper:
    def __init__(self, veh_num, node_num, human_choice, human_num, max_human_in_team, demand_penalty, time_penalty, time_limit, solver_time_limit, beta, flag_verbose = True):
        '''
        veh_num:            int, the agent number
        node_num:           int, the number of nodes,
                            it is assumed that 0 to node_num - 3 are point of interests,
                            node_num - 2 is the start,
                            node_num - 1 is the terminal,
                            and start and terminal are at the same location in the world
        human_choice:       int, the maximum number of POIs that a human can request to visit
        human_num:          int, the number of humans
        max_human_in_team:  int array of (veh_num, )
        demand_penalty:     float, default 1000.0
        time_penalty:       float, defalt 1.0
        time_limit:         float, default 500
        '''
        self.veh_num = veh_num                              # The number of agents (robots/vehicles/guides)
        self.node_num = node_num                            # The number of nodes (places of interest + 2). The "2" is due to a start and terminal node
        self.place_num = node_num - 2                       # The number of places of interest (POI)
        self.human_choice = human_choice                    # The maximum number of POIs that a human can request to visit
        self.human_num = human_num                          # The number of humans
        self.max_human_in_team = max_human_in_team + 0      # The maximum number of humans that a robot can guide
        self.demand_penalty = demand_penalty                # The penalty on dropping a human requested POI
        self.time_penalty = time_penalty                    # The penalty on the total time consumption of the tours
        self.time_limit = time_limit                        # The time limit on the tours
        self.flag_verbose = flag_verbose                    # If True, a lot of results will be printed to the consoles
        self.solver_time_limit = solver_time_limit          # The time limit for the solver
        self.beta = beta
        self.flag_initialize = 0                            # 0 means initialize the routes for the vehicles by solving a routing problem
                                                            # 1 means initialize the routes randomly

        self.evaluator = ResultEvaluator(veh_num, node_num, human_num, demand_penalty, time_penalty)


    def initialize_human_demand(self, human_demand_int = None):
        '''
        Inputs:
        human_demand_int: Let it be None
        ------------------------------------------------------
        Output:
        human_demand_bool: bool array of (human_num, place_num) indicating whether a human want to goto a place
        human_demand_int_unique: list of size (human_num, ), each list element is a int array indicating the place id that that human want to visit
        '''
        if human_demand_int is None:
            human_demand_int = np.random.randint(0, self.place_num, (self.human_num,self.human_choice) )
        temp_index = np.arange(self.human_num).reshape(-1,1).repeat(self.human_choice,axis=1)
        human_demand_bool = np.zeros((self.human_num, self.place_num), dtype=np.float64)
        human_demand_bool[temp_index.reshape(-1), human_demand_int.reshape(-1)] = 1.0
        human_demand_int_unique = []
        for l in range(self.human_num):
            human_demand_int_unique.append(np.unique(human_demand_int[l]))
        return human_demand_bool, human_demand_int_unique

    def plan(self, edge_time, node_time, edge_time_std, node_time_std, human_demand_bool, node_seq = None, max_iter = 10, flag_initialize = 0, flag_solver = 1):
        '''
        Inputs:
        edge_time:          float array of (veh_num, node_num, node_num)
        node_time:          float array of (veh_num, node_num)
        edge_time_std:      float array of (veh_num, node_num, node_num)
        node_time_std:      float array of (veh_num, node_num)
        human_demand_bool:  bool array of (human_num, place_num) indicating whether a human want to goto a place
        node_seq:           a sequence constraint, if no such constraints, let it be None
                            example [[0,1,2], [3,4]] means if a human want to visit 1, he/she must visit 0 first,
                                                           if a human want to visit 2, he/she must visit 1 first;
                                          apart from that, if a human want to visit 4, he/she must visit 3 first
        max_iter:           the maximum number of iteration for the optimization algorithm, set to 10 by default
        flag_initialize:    0 means initialize the routes for the vehicles by solving a routing problem
                            1 means initialize the routes randomly
        flag_solver:        0 means GUROBI-based exact bilinear solver
                            1 means Ortool-based heuristic solver
        ------------------------------------------------------
        Outputs:
        flag_success:       bool, whether the optimization is successful
        route_list:         list of size (veh_num, ), each element is a list of nodes
                            ignore the route_list[i][0] and route_list[i][-1], they are start and terminal nodes
                            example: [[10, 5, 0, 6, 10], [10, 3, 4, 5, 10], [10, 3, 4, 7, 10], [10, 3, 4, 6, 0, 10]]
        route_time_list:    list of size (veh_num, ), each element is a list of time
                            example: [[0, 89, 185, 277, 386], [0, 104, 161, 344, 433], [0, 104, 161, 205, 351], [0, 104, 161, 313, 405, 468]]
        team_list:          list of size (node_num-2, ), each element is a list of vehicle id that visit that node
                            example: [[0, 3], [], [], [1, 2, 3], [1, 2, 3], [0, 1], [0, 3], [2], [], []]
        human_in_team:      int list of size (human_num), each elements indicate which vehicle that human follows
        y_sol:              bool array of (veh_num, node_num-2), y_sol[k, i] means whether vehicle[k] visits node[i]
        z_sol:              bool array of (human_num, veh_num), z_sol[l, k] means whether human[l] follows vehicle[k]
        sum_obj_list:       float array of (max_iter, ), each element is the objective function value at that iteration
        demand_obj_list:    float array of (max_iter, ), each element is the number of dropped demand at that iteration
        result_max_time_list: float array of (max_iter, ), each element is the time usage of the whole guidance mission at that iteration
        '''
        flag_dict = True
        result_dict = {}
        start_time = time.time()
        if flag_solver == 1:
            # Heuristic solver using Ortool
            route_list, route_time_list, team_list, y_sol = self.initialize_plan(edge_time, node_time, flag_initialize)
            flag_success, route_list, route_time_list, team_list, human_in_team, y_sol, z_sol, sum_obj_list, demand_obj_list, result_max_time_list = self.generate_plan(edge_time, node_time, human_demand_bool, route_list, y_sol, node_seq, max_iter)
            sum_obj, demand_obj, result_sum_time, node_visit, obj_dict = self.evaluator.objective_fcn(edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool, flag_dict, self.beta, edge_time_std, node_time_std)
        else:
            # Exact solver using GUROBI
            routing_solver = GurobiRoutingSolver(self.veh_num, self.node_num, self.human_num, self.demand_penalty, self.time_penalty, self.time_limit, self.solver_time_limit, self.beta)
            routing_solver.set_model(edge_time, node_time)
            routing_solver.set_bilinear_model(edge_time, node_time, edge_time_std, node_time_std, human_demand_bool, self.max_human_in_team, node_seq)
            flag_success, temp_result_dict = routing_solver.optimize()
            route_list, route_time_list, team_list, y_sol, human_in_team, z_sol = routing_solver.get_plan(True)
            sum_obj, demand_obj, result_sum_time, node_visit, obj_dict = self.evaluator.objective_fcn(edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool, flag_dict, self.beta, edge_time_std, node_time_std)
            # Partially construct the result dictionary
            sum_obj_list = np.ones(2*max_iter, dtype=np.float64) * sum_obj
            demand_obj_list = np.ones(2*max_iter, dtype=np.float64) * demand_obj
            result_max_time_list = np.ones(2*max_iter, dtype=np.float64) * result_sum_time
            result_dict['lower_bound'] = temp_result_dict['ObjBound']
            result_dict['lin_obj'] = temp_result_dict['ObjVal']
            result_dict['optimality_gap'] = (sum_obj - result_dict['lower_bound']) / result_dict['lower_bound']
            result_dict['optimality_gap_wrong'] = (sum_obj - result_dict['lower_bound']) / sum_obj
            if edge_time_std is not None:
                result_dict['lin_cvar'] = (temp_result_dict['ObjVal'] - self.demand_penalty * demand_obj) / self.time_penalty
            else:
                result_dict['lin_cvar'] = 0.0
        end_time = time.time()
        optimization_time = end_time - start_time

        # Construct the result dictionary
        result_dict['optimization_time'] = optimization_time
        result_dict['sum_obj'] = sum_obj
        result_dict['demand_obj'] = demand_obj
        result_dict['sum_obj_list'] = sum_obj_list
        result_dict['demand_obj_list'] = demand_obj_list
        result_dict['result_max_time_list'] = result_max_time_list
        result_dict['total_demand'] = human_demand_bool.sum()
        result_dict['dropped_demand_rate'] = demand_obj / result_dict['total_demand']
        result_dict['optimization_time'] = optimization_time
        result_dict['result_time_list'] = obj_dict['result_time_list']
        result_dict['result_max_time'] = obj_dict['result_max_time']
        result_dict['result_sum_time'] = obj_dict['result_sum_time']
        result_dict['result_time_cvar'] = obj_dict['result_time_cvar']
        result_dict['result_max_time_cvar'] = obj_dict['result_max_time_cvar']
        result_dict['result_sum_time_cvar'] = obj_dict['result_sum_time_cvar']
        # Print
        print('\n')
        for key, value in result_dict.items():
            print(key, ':', value)
        print('\n')
        print('optimization_time = ', optimization_time)
        return flag_success, route_list, route_time_list, team_list, human_in_team, y_sol, z_sol, result_dict

    def initialize_plan(self, edge_time, node_time, flag_initialize = 0):
        '''
        Inputs:
        edge_time:          float array of (veh_num, node_num, node_num)
        node_time:          float array of (veh_num, node_num)
        flag_initialize:    0 means initialize the routes for the vehicles by solving a routing problem
                            1 means initialize the routes randomly
        ------------------------------------------------------
        Outputs:
        route_list:         list of size (veh_num, ), each element is a list of nodes
                            ignore the route_list[i][0] and route_list[i][-1], they are start and terminal nodes
                            example: [[10, 5, 0, 6, 10], [10, 3, 4, 5, 10], [10, 3, 4, 7, 10], [10, 3, 4, 6, 0, 10]]
        route_time_list:    list of size (veh_num, ), each element is a list of time
                            example: [[0, 89, 185, 277, 386], [0, 104, 161, 344, 433], [0, 104, 161, 205, 351], [0, 104, 161, 313, 405, 468]]
        team_list:          list of size (node_num-2, ), each element is a list of vehicle id that visit that node
                            example: [[0, 3], [], [], [1, 2, 3], [1, 2, 3], [0, 1], [0, 3], [2], [], []]
        y_sol:              bool array of (veh_num, node_num-2), y_sol[k, i] means whether a vehicle visits node i
        '''
        # Initialize an routing plan
        routing_solver = OrtoolRoutingSolver(self.veh_num, self.node_num, self.human_num, self.demand_penalty, self.time_penalty, self.time_limit, self.solver_time_limit)
        self.flag_initialize = flag_initialize
        if flag_initialize == 0:
            routing_solver.set_model(edge_time, node_time)
            routing_solver.optimize(self.flag_verbose)
            route_list, route_time_list, team_list, y_sol = routing_solver.get_plan()
        else:
            route_list, route_time_list, team_list, y_sol = routing_solver.get_random_plan(edge_time, node_time)
        return route_list, route_time_list, team_list, y_sol


    def generate_plan(self, edge_time, node_time, human_demand_bool, route_list_initial, y_sol_inital, node_seq = None, max_iter = 10):
        '''
        Inputs:
        edge_time:          float array of (veh_num, node_num, node_num)
        node_time:          float array of (veh_num, node_num)
        human_demand_bool:  bool array of (human_num, place_num) indicating whether a human want to goto a place
        route_list_initial: an initial guess of route_list from function initialize_plan()
        y_sol_inital:       an initial guess of y_sol from function initialize_plan()
        node_seq:           a sequence constraint, if no such constraints, let it be None
                            example [[0,1,2], [3,4]] means if a human want to visit 1, he/she must visit 0 first,
                                                           if a human want to visit 2, he/she must visit 1 first;
                                          apart from that, if a human want to visit 4, he/she must visit 3 first
        max_iter:           the maximum number of iteration for the optimization algorithm, set to 10 by default
        ------------------------------------------------------
        Outputs:
        flag_success:       bool, whether the optimization is successful
        route_list:         list of size (veh_num, ), each element is a list of nodes
                            ignore the route_list[i][0] and route_list[i][-1], they are start and terminal nodes
                            example: [[10, 5, 0, 6, 10], [10, 3, 4, 5, 10], [10, 3, 4, 7, 10], [10, 3, 4, 6, 0, 10]]
        route_time_list:    list of size (veh_num, ), each element is a list of time
                            example: [[0, 89, 185, 277, 386], [0, 104, 161, 344, 433], [0, 104, 161, 205, 351], [0, 104, 161, 313, 405, 468]]
        team_list:          list of size (node_num-2, ), each element is a list of vehicle id that visit that node
                            example: [[0, 3], [], [], [1, 2, 3], [1, 2, 3], [0, 1], [0, 3], [2], [], []]
        human_in_team:      int list of size (human_num), each elements indicate which vehicle that human follows 
        y_sol:              bool array of (veh_num, node_num-2), y_sol[k, i] means whether vehicle[k] visits node[i]
        z_sol:              bool array of (human_num, veh_num), z_sol[l, k] means whether human[l] follows vehicle[k]
        sum_obj_list:       float array of (max_iter, ), each element is the objective function value at that iteration
        demand_obj_list:    float array of (max_iter, ), each element is the number of dropped demand at that iteration
        result_max_time_list: float array of (max_iter, ), each element is the time usage of the whole guidance mission at that iteration
        '''
        # Run the large neighborhood search (LNS) algorithm
        # Initialize some lists to store intermediate results
        route_list = route_list_initial # Shallow copy
        sum_obj_list = np.empty(2*max_iter, dtype=np.float64)
        demand_obj_list = np.empty(2*max_iter, dtype=np.float64)
        result_max_time_list = np.empty(2*max_iter, dtype=np.float64)

        # Initialize the solvers for the routing and matching problems
        routing_solver = OrtoolRoutingSolver(self.veh_num, self.node_num, self.human_num, self.demand_penalty, self.time_penalty, self.time_limit, self.solver_time_limit)
        human_matcher = OrtoolHumanMatcher(self.human_num, self.veh_num, self.max_human_in_team)

        # Set the intial value
        y_sol = y_sol_inital + 0
        z_sol = None
        if self.flag_verbose:
            # Print the intial objective function value
            sum_obj, demand_obj, result_max_time, node_visit = self.evaluator.objective_fcn(edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool)
            print('sum_obj = demand_penalty * demand_obj + time_penalty * max_time = %f * %f + %f * %f = %f' % (self.demand_penalty, demand_obj, self.time_penalty, result_max_time, sum_obj))
        # Start iterating
        for i_iter in range(max_iter):
            # Solve the matching problem
            temp_flag_success, human_in_team, z_sol, demand_result = human_matcher.optimize(human_demand_bool, y_sol)
            if not temp_flag_success:
                break
            # Store intermediate results
            sum_obj, demand_obj, result_max_time, node_visit = self.evaluator.objective_fcn(edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool)
            if self.flag_verbose:
                print('sum_obj1 = demand_penalty * demand_obj + time_penalty * max_time = %f * %f + %f * %f = %f ... (1)' % (self.demand_penalty, demand_obj, self.time_penalty, result_max_time, sum_obj))
            sum_obj_list[2*i_iter] = sum_obj
            demand_obj_list[2*i_iter] = demand_obj
            result_max_time_list[2*i_iter] = result_max_time

            if (self.flag_initialize != 0) and (i_iter == 0):
                # if flag_initialize != 0, it means there is no initial routes, therefore, set to None
                route_list = None
            # Optimize the routing problem
            temp_flag_success, result_dict = routing_solver.optimize_sub(edge_time, node_time, z_sol, human_demand_bool, node_seq, route_list)
            if not temp_flag_success:
                break
            route_list, route_time_list, team_list, y_sol = routing_solver.get_plan(flag_sub_solver=True)

            # Store intermediate results
            sum_obj, demand_obj, result_max_time, node_visit = self.evaluator.objective_fcn(edge_time, node_time, route_list, z_sol, y_sol, human_demand_bool)
            if self.flag_verbose:
                print('sum_obj2 = demand_penalty * demand_obj + time_penalty * max_time = %f * %f + %f * %f = %f' % (self.demand_penalty, demand_obj, self.time_penalty, result_max_time, sum_obj))
            sum_obj_list[2*i_iter+1] = sum_obj
            demand_obj_list[2*i_iter+1] = demand_obj
            result_max_time_list[2*i_iter+1] = result_max_time

        # Check whether the optimization is successful
        flag_success = i_iter >= 1 # TODO: This condition is just a placeholder, not the actual condition
        return flag_success, route_list, route_time_list, team_list, human_in_team, y_sol, z_sol, sum_obj_list, demand_obj_list, result_max_time_list
