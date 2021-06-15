from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time

class OrtoolRoutingSolver:
    def __init__(self, veh_num, node_num, human_num, demand_penalty, time_penalty, flag_solver_type):
        self.LARGETIME = 1000.0
        self.veh_num = veh_num
        self.node_num = node_num
        self.human_num = human_num
        # self.demand_penalty = demand_penalty
        self.time_penalty = time_penalty

        self.start_node = self.node_num - 2
        self.global_penalty = 1000.0

        # Create Routing Model.
        self.manager = pywrapcp.RoutingIndexManager(self.node_num-1, self.veh_num, self.start_node)
        self.solver = pywrapcp.RoutingModel(self.manager)
        self.solution = None

    def set_model(self, edge_time, node_time):
        distance_matrix = edge_time[0, :self.node_num-1, :self.node_num-1] + 0
        distance_matrix += node_time[0, :self.node_num-1].reshape(self.node_num-1, 1)

        self.data = {}
        self.data['edge_time'] = edge_time
        self.data['node_time'] = node_time
        self.data['distance_matrix'] = distance_matrix

        print('distance_matrix')

        transit_callback_index = self.solver.RegisterTransitCallback(self.distance_callback)

        # Define cost of each arc.
        self.solver.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = 'Time'
        self.solver.AddDimension(
            transit_callback_index,
            0,  # no slack
            3000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = self.solver.GetDimensionOrDie(dimension_name)
        temp_penalty = int(self.global_penalty * self.time_penalty)
        distance_dimension.SetGlobalSpanCostCoefficient(temp_penalty)
    
    def optimize(self):
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        # Solve the problem.
        start_time = time.time()
        self.solution = self.solver.SolveWithParameters(search_parameters)
        end_time = time.time()

        result_dict = {}
        result_dict['Optimized'] = self.solver.status() == 1
        result_dict['Status'] = self.solver.status()
        result_dict['Runtime'] = end_time - start_time
        # result_dict['IterCount'] = self.solver.iterations()
        # result_dict['NodeCount'] = self.solver.nodes()
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


    def get_plan(self, flag_verbose = False):
        """Prints solution on console."""
        route_node_list = []
        route_time_list = []
        team_list = [[] for i in range(self.node_num-2)]

        solution = self.solution
        if flag_verbose:
            print(f'Objective: {solution.ObjectiveValue()}')
        time_dimension = self.solver.GetDimensionOrDie('Time')
        total_max_time = 0
        for vehicle_id in range(self.veh_num):
            route_node = []
            route_time = []
            index = self.solver.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            while not self.solver.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                node_id = self.manager.IndexToNode(index)
                temp_min_time = solution.Min(time_var)
                temp_max_time = solution.Max(time_var)
                plan_output += '{0} Time({1},{2}) -> '.format(node_id, temp_min_time, temp_max_time)
                index = solution.Value(self.solver.NextVar(index))
                route_node.append(node_id)
                route_time.append(temp_min_time)
                if node_id != self.start_node:
                    team_list[node_id].append(vehicle_id)
            time_var = time_dimension.CumulVar(index)
            node_id = self.manager.IndexToNode(index)
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

        # print(route_node_list)
        # print(route_time_list)
        # print(team_list)
        return route_node_list, route_time_list, team_list


