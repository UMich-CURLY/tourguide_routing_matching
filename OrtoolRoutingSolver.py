import numpy as np
from numpy.lib.function_base import place
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time

class OrtoolRoutingSolver:
    def __init__(self, veh_num, node_num, human_num, demand_penalty, time_penalty, time_limit, solver_time_limit = 20):
        self.LARGETIME = 1000.0                 # A large number that should be larger than any possible time appeared in the optimization problem
        self.veh_num = veh_num                  # The number of agents (robots/vehicles/guides)
        self.node_num = node_num                # The number of nodes
        self.human_num = human_num              # The number of human
        self.demand_penalty = demand_penalty    # The penalty on dropping a POI
        self.time_penalty = time_penalty        # The penalty on the total time consumption of the tour
        if time_limit <= 1:                     # The time limit of the tours, integer valued
            self.time_limit = 300000
        else:
            self.time_limit = int(time_limit)

        self.start_node = self.node_num - 2     # Keep it, the start node is assumed to be No. node_num-2
        self.global_penalty = 1.0               # Keep this constant
        self.solver_time_limit = int(solver_time_limit)     # Time limit for the solver

    def optimize_sub(self, edge_time, node_time, z_sol, human_demand_bool, node_seq, route_list = None, flag_verbose = False):
        '''
        Optimize the routing problem
        ------------------------------------------------------
        z_sol:             (human_num, veh_num)
        human_demand_bool: (human_num, place_num), i.e. (human_num, node_num - 2)
        '''
        place_num = self.node_num-2
        penalty_mat = np.zeros((self.veh_num, place_num), dtype=np.float64)

        result_dict = {}
        result_dict['Optimized'] = True
        result_dict['Status'] = []
        start_time = time.time()

        # Create sub-routing model
        self.sub_manager = []
        self.sub_solver = []
        self.sub_solution = []
        for i in range(self.veh_num):
            a_sub_manager = pywrapcp.RoutingIndexManager(self.node_num-1, 1, self.start_node)
            a_sub_solver = pywrapcp.RoutingModel(a_sub_manager)
            self.sub_manager.append(a_sub_manager)
            self.sub_solver.append(a_sub_solver)
            self.sub_solution.append(None)

        for k in range(self.veh_num):
            for i in range(place_num):
                penalty_mat[k, i] = (z_sol[:, k] * human_demand_bool[:, i]).sum()
        # print('penalty_mat = ', penalty_mat)
        for k in range(self.veh_num):
            def temp_distance_callback(from_index, to_index):
                """Returns the distance between the two nodes."""
                # Convert from routing variable Index to distance matrix NodeIndex.
                from_node = self.sub_manager[k].IndexToNode(from_index)
                to_node = self.sub_manager[k].IndexToNode(to_index)
                if from_node == to_node:
                    dist_out = 0.0
                else:
                    dist_out = edge_time[k,from_node,to_node] + node_time[k,from_node]
                return dist_out

            transit_callback_index = self.sub_solver[k].RegisterTransitCallback(temp_distance_callback)

            # Define cost of each arc.
            self.sub_solver[k].SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            # Add Distance constraint.
            dimension_name = 'Time'
            self.sub_solver[k].AddDimension(
                transit_callback_index,
                0,  # no slack
                self.time_limit,  # vehicle maximum travel distance
                True,  # start cumul to zero
                dimension_name)
            # distance_dimension = self.sub_solver[k].GetDimensionOrDie(dimension_name)
            # temp_penalty = int(self.global_penalty * self.time_penalty)
            # distance_dimension.SetGlobalSpanCostCoefficient(temp_penalty)

            # Allow to drop nodes.
            for i in range(place_num):
                # temp_penalty = int(penalty_mat[k, i] * self.demand_penalty)
                temp_penalty = int(penalty_mat[k, i] * self.demand_penalty * self.global_penalty)
                self.sub_solver[k].AddDisjunction([self.sub_manager[k].NodeToIndex(i)], temp_penalty)

            # Add sequence constraints, see the references in README.md
            if node_seq is not None:
                self.add_seq_constraint(self.sub_solver[k], self.sub_manager[k], node_seq)

            # Solve the problem.
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.time_limit.seconds = self.solver_time_limit
            if (route_list is not None) and len(route_list[k]) > 2:
                initial_solution = self.sub_solver[k].ReadAssignmentFromRoutes([route_list[k][1:-1]], True)
                a_sub_solution = self.sub_solver[k].SolveFromAssignmentWithParameters(initial_solution, search_parameters)
            else:
                search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
                a_sub_solution = self.sub_solver[k].SolveWithParameters(search_parameters)

            # Construct the result dictionary
            result_dict['Status'].append(self.sub_solver[k].status())
            if self.sub_solver[k].status() != 1:
                result_dict['Optimized'] = False
                continue
            self.sub_solution[k] = a_sub_solution

        end_time = time.time()
        result_dict['Runtime'] = end_time - start_time

        # result_dict['IterCount'] = self.solver.iterations()
        # result_dict['NodeCount'] = self.solver.nodes()
        if flag_verbose:
            print('Solution found: %d' % result_dict['Optimized'])
            print('Optimization status:', result_dict['Status'])
            print('Problem solved in %f seconds' % result_dict['Runtime'])
        flag_success = result_dict['Optimized']
        return flag_success, result_dict

    def set_model(self, edge_time, node_time, node_seq = None):
        # Create Routing Model.
        self.manager = pywrapcp.RoutingIndexManager(self.node_num-1, self.veh_num, self.start_node)
        self.solver = pywrapcp.RoutingModel(self.manager)
        self.solution = None

        distance_matrix = edge_time[0, :self.node_num-1, :self.node_num-1] + 0
        distance_matrix += node_time[0, :self.node_num-1].reshape(self.node_num-1, 1)

        self.data = {}
        self.data['edge_time'] = edge_time
        self.data['node_time'] = node_time
        self.data['distance_matrix'] = distance_matrix

        transit_callback_index = self.solver.RegisterTransitCallback(self.distance_callback)

        # Define cost of each arc.
        self.solver.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = 'Time'
        self.solver.AddDimension(
            transit_callback_index,
            0,  # no slack
            300000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = self.solver.GetDimensionOrDie(dimension_name)
        temp_penalty = int(self.global_penalty * self.time_penalty)
        distance_dimension.SetGlobalSpanCostCoefficient(temp_penalty)

    def add_seq_constraint(self, solver, manager, node_seq):
        distance_dimension = solver.GetDimensionOrDie('Time')
        for i_seq in range(len(node_seq)):
            for i_node in range(len(node_seq[i_seq]) - 1):
                node_i = node_seq[i_seq][i_node]
                node_j = node_seq[i_seq][i_node+1]
                nodeid_i = manager.NodeToIndex(node_i)
                nodeid_j = manager.NodeToIndex(node_j)
                # print('node:', node_i, node_j, 'index:', nodeid_i, nodeid_j)
                # solver.AddPickupAndDelivery(nodeid_i, nodeid_j)
                # solver.solver().Add(solver.VehicleVar(nodeid_i) == solver.VehicleVar(nodeid_j))
                # solver.solver().Add(distance_dimension.CumulVar(nodeid_i) <= distance_dimension.CumulVar(nodeid_j))

                # j active is based on i active
                solver.solver().Add(solver.ActiveVar(nodeid_j) <= solver.ActiveVar(nodeid_i))
                # j's time is after i's time (visit i before j)
                solver.solver().Add(distance_dimension.CumulVar(nodeid_i) <= distance_dimension.CumulVar(nodeid_j))
                # i and j should be using the same vehicle
                constraintActive = solver.ActiveVar(nodeid_i) * solver.ActiveVar(nodeid_j)
                solver.solver().Add(constraintActive * (solver.VehicleVar(nodeid_i) - solver.VehicleVar(nodeid_j)) == 0 )

    def optimize(self, flag_verbose = False):
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = self.solver_time_limit
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        start_time = time.time()
        self.solution = self.solver.SolveWithParameters(search_parameters)
        end_time = time.time()

        # Construct the result dictionary
        result_dict = {}
        result_dict['Optimized'] = self.solver.status() == 1
        result_dict['Status'] = self.solver.status()
        result_dict['Runtime'] = end_time - start_time
        # result_dict['IterCount'] = self.solver.iterations()
        # result_dict['NodeCount'] = self.solver.nodes()

        # Print the results
        if flag_verbose:
            print('Solution found: %d' % result_dict['Optimized'])
            print('Optimization status: %d' % result_dict['Status'])
            print('Problem solved in %f seconds' % result_dict['Runtime'])
            # print('Problem solved in %d iterations' % result_dict['IterCount'])
            # print('Problem solved in %d branch-and-bound nodes' % result_dict['NodeCount'])
        return result_dict

    def distance_callback(self, from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        if from_node == to_node:
            dist_out = 0.0
        else:
            # dist_out = self.data['distance_matrix'][from_node][to_node]
            dist_out = self.data['edge_time'][0,from_node,to_node] + self.data['node_time'][0,from_node]
        # print('dist_out = ', dist_out)
        return dist_out

    def get_random_plan(self, edge_time, node_time):
        # Initialize a random routing plan
        place_num = self.node_num-2
        place_perm = np.random.permutation(place_num)
        route_node_list = []
        team_list = [[] for i in range(self.node_num-2)]
        y_sol = np.zeros((self.veh_num, self.node_num-2), dtype=np.float64)
        for i_place in range(place_num):
            i_veh = i_place % self.veh_num
            node_id = place_perm[i_place]
            if i_place == i_veh:
                route_node_list.append([self.start_node])
            route_node_list[i_veh].append(node_id)
            team_list[node_id].append(i_veh)
            y_sol[i_veh, node_id] = 1.0
        for i_veh in range(self.veh_num):
            route_node_list[i_veh].append(self.start_node)
        route_time_list = []
        for i_veh in range(self.veh_num):
            assert len(route_node_list) > 2, 'OrtoolRoutingSolver.get_random_plan: An empty route!'
            route_time = 0.0
            route_time_list.append([route_time])
            for i_node in range(len(route_node_list[i_veh]) - 1):
                node_i = route_node_list[i_veh][i_node]
                node_j = route_node_list[i_veh][i_node+1]
                route_time += edge_time[i_veh,node_i,node_j] + node_time[i_veh,node_i]
                route_time_list[i_veh].append(route_time)
        return route_node_list, route_time_list, team_list, y_sol

    def get_plan(self, flag_sub_solver = False, flag_verbose = False):
        # Output the plans
        """Prints solution on console."""
        route_node_list = []
        route_time_list = []
        team_list = [[] for i in range(self.node_num-2)]

        if not flag_sub_solver:
            solution = self.solution
            solver = self.solver
            manager = self.manager
            time_dimension = solver.GetDimensionOrDie('Time')
            if flag_verbose:
                print(f'Objective: {solution.ObjectiveValue()}')
        total_max_time = 0
        for vehicle_id in range(self.veh_num):
            route_node = []
            route_time = []
            if flag_sub_solver:
                solution = self.sub_solution[vehicle_id]
                solver = self.sub_solver[vehicle_id]
                manager = self.sub_manager[vehicle_id]
                index = solver.Start(0)
                time_dimension = solver.GetDimensionOrDie('Time')
            else:
                index = solver.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            while not solver.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                node_id = manager.IndexToNode(index)
                temp_min_time = solution.Min(time_var)
                temp_max_time = solution.Max(time_var)
                plan_output += '{0} Time({1},{2}) -> '.format(node_id, temp_min_time, temp_max_time)
                index = solution.Value(solver.NextVar(index))
                route_node.append(node_id)
                route_time.append(temp_min_time)
                if node_id < self.start_node:
                    team_list[node_id].append(vehicle_id)
            time_var = time_dimension.CumulVar(index)
            node_id = manager.IndexToNode(index)
            temp_min_time = solution.Min(time_var)
            temp_max_time = solution.Max(time_var)
            plan_output += '{0} Time({1},{2})\n'.format(node_id, temp_min_time, temp_max_time)
            plan_output += 'Time of the route: {}min\n'.format(temp_min_time)
            if temp_min_time > total_max_time:
                total_max_time = temp_min_time
            route_node.append(node_id)
            route_time.append(temp_min_time)
            route_node_list.append(route_node)
            route_time_list.append(route_time)

            if flag_verbose:
                print(plan_output)
                print('time_var = ', temp_min_time)
        if flag_verbose:
            print('Max time of all routes: {}min'.format(total_max_time))

        y_sol = np.zeros((self.veh_num, self.node_num-2), dtype=np.float64)
        for i in range(self.node_num-2):
            for k in team_list[i]:
                y_sol[k,i] = 1.0
        # print(route_node_list)
        # print(route_time_list)
        # print(team_list)
        return route_node_list, route_time_list, team_list, y_sol
