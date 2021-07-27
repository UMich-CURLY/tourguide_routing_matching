from typing import Sequence
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class GurobiRoutingSolver:
    def __init__(self, veh_num, node_num, human_num, demand_penalty, time_penalty, time_limit, solver_time_limit = 20.0, beta = 0.8):
        self.LARGETIME = 1000.0
        self.veh_num = veh_num
        self.node_num = node_num
        self.human_num = human_num
        self.demand_penalty = demand_penalty
        self.time_penalty = time_penalty
        if time_limit <= 1:
            self.time_limit = 30000
        else:
            self.time_limit = time_limit

        self.start_node = self.node_num - 2
        self.end_node = self.node_num - 1

        self.solver = gp.Model("Routing")
        self.solver.Params.timeLimit = solver_time_limit
        self.x_var = self.solver.addVars(self.veh_num, self.node_num, self.node_num, vtype=GRB.BINARY, name='x')
        self.y_var = self.solver.addVars(self.veh_num, self.node_num-1, vtype=GRB.BINARY, name='y')
        self.time_var = self.solver.addVars(self.veh_num, self.node_num, vtype=GRB.CONTINUOUS, name='t', lb=0.0, ub=self.LARGETIME)
        self.max_time_var = self.solver.addVar(0.0, self.LARGETIME, 1.0, GRB.CONTINUOUS, "t_max")
        self.z_var = self.solver.addVars(self.human_num, self.veh_num, vtype=GRB.BINARY, name='z')
        self.alpha_var = None
        self.w_var = None

        self.flag_time_lifting = True
        self.sample_num = 100
        self.beta = beta

    def optimize(self):
        self.solver.optimize()
        result_dict = {}
        result_dict['Status'] = self.solver.getAttr('Status')
        result_dict['Runtime'] = self.solver.getAttr('Runtime')
        result_dict['IterCount'] = self.solver.getAttr('IterCount')
        result_dict['NodeCount'] = self.solver.getAttr('NodeCount')
        print('Optimization status: %d' % result_dict['Status'])
        print('Problem solved in %f seconds' % result_dict['Runtime'])
        print('Problem solved in %d iterations' % result_dict['IterCount'])
        print('Problem solved in %d branch-and-bound nodes' % result_dict['NodeCount'])
        flag_success = (result_dict['Status'] == 2) or (result_dict['Status'] >= 7) # https://www.gurobi.com/documentation/9.1/refman/optimization_status_codes.html
        return flag_success, result_dict

    def set_bilinear_model(self, edge_time, node_time, edge_time_std, node_time_std, human_demand_bool, max_human_in_team, node_seq):
        obj = 0.0
        flag_uncertainty = (edge_time_std is not None) and (node_time_std is not None)
        # Objective function: time part
        if flag_uncertainty:
            assert edge_time.shape == edge_time_std.shape, 'edge_time.shape == edge_time_std.shape not satisfied'
            assert node_time.shape == node_time_std.shape, 'node_time.shape == node_time_std.shape not satisfied'
            self.alpha_var = self.solver.addVars(self.veh_num, vtype=GRB.CONTINUOUS, name='alpha', lb=0.0, ub=self.LARGETIME)
            self.w_var = self.solver.addVars(self.veh_num, self.sample_num, vtype=GRB.CONTINUOUS, name='w', lb=0.0, ub=self.LARGETIME)
            temp_coeff = 1.0 / (self.sample_num * (1 - self.beta))
            for k in range(self.veh_num):
                obj += self.time_penalty * self.alpha_var[k]
                for i_sample in range(self.sample_num):
                    obj += self.time_penalty * temp_coeff * self.w_var[k, i_sample]
        else:
            if self.flag_time_lifting:
                for k in range(self.veh_num):
                    for i in range(self.node_num-1):
                        obj += self.time_penalty * node_time[k, i] * self.y_var[k, i]
                        for j in range(self.node_num):
                            obj += self.time_penalty * edge_time[k, i, j] * self.x_var[k, i, j]
            else:
                for k in range(self.veh_num):
                    obj += self.time_penalty * self.time_var[k, self.end_node]
        # Objective function: demand part
        for l in range(self.human_num):
            for k in range(self.veh_num):
                for i in range(self.node_num-2):
                    obj += self.demand_penalty * human_demand_bool[l, i] * self.z_var[l, k] * (1.0 - self.y_var[k, i])
        self.solver.setObjective(obj, GRB.MINIMIZE)

        self.set_bilinear_constraint(edge_time, node_time, max_human_in_team)
        # Sequence constraints
        if node_seq is not None:
            self.add_seq_constraint(node_seq)
        if flag_uncertainty:
            self.set_cvar_constraint(edge_time, node_time, edge_time_std, node_time_std)

    def set_cvar_constraint(self, edge_time, node_time, edge_time_std, node_time_std):
        for i_sample in range(self.sample_num):
            temp_edge_time = edge_time + np.random.randn(*edge_time.shape) * edge_time_std
            temp_node_time = node_time + np.random.randn(*node_time.shape) * node_time_std
            temp_edge_time[temp_edge_time < 0] = 0
            temp_node_time[temp_node_time < 0] = 0
            for k in range(self.veh_num):
                constr = - self.alpha_var[k]
                for i in range(self.node_num-1):
                    constr += temp_node_time[k, i] * self.y_var[k, i]
                    for j in range(self.node_num):
                        constr += temp_edge_time[k, i, j] * self.x_var[k, i, j]
                constr_name = 'cvar[' + str(k) + ',' + str(i_sample) + ']'
                self.solver.addConstr(self.w_var[k, i_sample] >= constr, constr_name)

    def set_bilinear_constraint(self, edge_time, node_time, max_human_in_team):
        # Human assignment constraints
        for l in range(self.human_num):
            constr = 0
            for k in range(self.veh_num):
                constr += self.z_var[l, k]
            constr_name = 'human_assign[' + str(l) + ',' + str(k) + ']'
            self.solver.addConstr(constr == 1, constr_name)
        # Team size limit constraints
        for k in range(self.veh_num):
            constr = 0
            for l in range(self.human_num):
                constr += self.z_var[l, k]
            constr_name = 'max_human[' + str(l) + ',' + str(k) + ']'
            self.solver.addConstr(constr <= max_human_in_team[k], constr_name)
        # Time limit constraints
        if self.flag_time_lifting:
            for k in range(self.veh_num):
                constr = 0
                for i in range(self.node_num-1):
                    constr += node_time[k, i] * self.y_var[k, i]
                    for j in range(self.node_num):
                        constr += edge_time[k, i, j] * self.x_var[k, i, j]
                constr_name = 'time_limit[' + str(k) + ']'
                self.solver.addConstr(constr <= self.time_limit, constr_name)
        else:
            for k in range(self.veh_num):
                constr = self.time_var[k, self.end_node]
                constr_name = 'time_limit[' + str(k) + ']'
                self.solver.addConstr(constr <= self.time_limit, constr_name)

    def add_seq_constraint(self, node_seq):
        for i_seq in range(len(node_seq)):
            for i_node in range(len(node_seq[i_seq]) - 1):
                node_i = node_seq[i_seq][i_node]
                node_j = node_seq[i_seq][i_node+1]
                for k in range(self.veh_num):
                    constr_name = 'seq_depend[' + str(k) + ',' + str(node_i) + ',' + str(node_j) + ']'
                    self.solver.addConstr(self.y_var[k, node_i] >= self.y_var[k, node_j], constr_name)
                    constr_name = 'seq_time[' + str(k) + ',' + str(node_i) + ',' + str(node_j) + ']'
                    self.solver.addConstr(self.time_var[k, node_i] <= self.time_var[k, node_j] + self.LARGETIME * (1 - self.y_var[k, node_j]), constr_name)

    def set_objective(self):
        obj = self.time_penalty * self.max_time_var
        self.solver.setObjective(obj, GRB.MINIMIZE)

    def set_all_task_complete(self):
        for i in range(self.node_num-2):
            constr = 0
            for k in range(self.veh_num):
                constr += self.y_var[k, i]
            constr_name = 'task_complete[' + str(i) + ']'
            self.solver.addConstr(constr >= 1, constr_name)

    def set_model(self, edge_time, node_time):
        '''
        edge_time: (veh_num, node_num, node_num)
        node_time: (veh_num, node_num)
        '''
        # Network flow constraints
        for k in range(self.veh_num):
            for i in range(self.node_num):
                # From node i to i is not allowed
                constr_name = 'self[' + str(k) + ',' + str(i) + ']'
                self.solver.addConstr(self.x_var[k,i,i] == 0, constr_name)
                # From node i to start is not allowed
                constr_name = 'start[' + str(k) + ',' + str(i) + ']'
                self.solver.addConstr(self.x_var[k,i,self.start_node] == 0, constr_name)
                # From end to node i is not allowed
                constr_name = 'end[' + str(k) + ',' + str(i) + ']'
                self.solver.addConstr(self.x_var[k,self.end_node,i] == 0, constr_name)
                # incoming edge <= 1
                constr = 0
                for j in range(self.node_num):
                    constr += self.x_var[k,j,i]
                constr_name = 'flow_less_one[' + str(k) + ',' + str(i) + ']'
                self.solver.addConstr(constr <= 1, constr_name)
                if i >= self.start_node:
                    continue
                # incoming edge == outgoing edge
                constr = 0
                for j in range(self.node_num):
                    constr += self.x_var[k,i,j] - self.x_var[k,j,i]
                constr_name = 'flow[' + str(k) + ',' + str(i) + ']'
                self.solver.addConstr(constr == 0, constr_name)
            # From start to all <= 1
            constr = 0
            for i in range(self.node_num):
                constr += self.x_var[k,self.start_node,i]
            constr_name = 'flow_less_one[' + str(k) + ',' + 's' + ']'
            self.solver.addConstr(constr <= 1, constr_name)
        
        # Task constraint
        for k in range(self.veh_num):
            constr1 = 0
            for j in range(self.node_num):
                constr1 += self.x_var[k,self.start_node,j]
            for i in range(self.node_num-1):
                # Variable Relationship Constraint: Outgoing edges == y
                constr = 0
                for j in range(self.node_num):
                    constr += self.x_var[k,i,j]
                constr_name = 'task[' + str(k) + ',' + str(i) + ']'
                self.solver.addConstr(constr == self.y_var[k,i], constr_name)
                # Flow Constraint: Incoming edge <= outgoing edge from start (redundant but useful for solver)
                constr_name1 = 'task<=start[' + str(k) + ',' + str(i) + ']'
                self.solver.addConstr(constr <= constr1, constr_name1)
        
        # Time constraint
        for k in range(self.veh_num):
            # Start time is zero
            constr_name = 'start_time[' + str(k) + ']'
            self.solver.addConstr(self.time_var[k,self.start_node] == 0, constr_name)
            for i in range(self.node_num):
                for j in range(self.node_num):
                    constr = self.time_var[k,i] - self.time_var[k,j] + edge_time[k,i,j] + node_time[k,i] - self.LARGETIME * (1 - self.x_var[k,i,j])
                    constr_name = 'edge_time[' + str(k) + ',' + str(i) + ',' + str(j) + ']'
                    self.solver.addConstr(constr <= 0, constr_name)
                    # Optional
                    constr = self.time_var[k,i] - self.time_var[k,j] + edge_time[k,i,j] + node_time[k,i] + self.LARGETIME * (1 - self.x_var[k,i,j])
                    constr_name = 'edge_time2[' + str(k) + ',' + str(i) + ',' + str(j) + ']'
                    self.solver.addConstr(constr >= 0, constr_name)
        for k in range(self.veh_num):
            for i in range(self.node_num):
                constr_name = 'max_time[' + str(k) + ',' + str(i) + ']'
                self.solver.addConstr(self.time_var[k,i] <= self.max_time_var, constr_name)


        # Energy constraint
        # Placeholder

    def get_plan(self, flag_bilinear = False):
        time_mat = np.zeros((self.veh_num, self.node_num), dtype=np.float64)
        for k in range(self.veh_num):
            for i in range(self.node_num):
                time_mat[k,i] = self.time_var[k,i].x

        route_node_list = []
        route_time_list = []
        for k in range(self.veh_num):
            curr_node = self.start_node
            route_node = [curr_node]
            route_time = [time_mat[k,curr_node]]
            for ii in range(100):
                next_node = -1
                for i in range(self.node_num):
                    if self.x_var[k,curr_node,i].x > 0.5:
                        next_node = i
                        break
                if next_node == -1:
                    break
                route_node.append(next_node)
                route_time.append(time_mat[k,next_node])
                curr_node = next_node
            route_node_list.append(route_node)
            route_time_list.append(route_time)

        # for k in [1]:
        #     for i in range(self.node_num):
        #         print(self.x_var[k,i,5].varName, self.x_var[k,i,5].x)
        #     for i in range(self.node_num):
        #         print(self.x_var[k,i,2].varName, self.x_var[k,i,2].x)

        team_list = []
        for i in range(self.node_num-2):
            team = []
            for k in range(self.veh_num):
                if self.y_var[k,i].x > 0.5:
                    team.append(k)
            team_list.append(team)
            print(team)

        y_sol = np.zeros((self.veh_num, self.node_num-2), dtype=np.float64)
        for i in range(self.node_num-2):
            for k in team_list[i]:
                y_sol[k,i] = 1.0

        # print('route_time_list = ', route_time_list)
        if not flag_bilinear:
            return route_node_list, route_time_list, team_list, y_sol
        
        z_sol = np.zeros((self.human_num, self.veh_num), dtype=np.float64)
        human_in_team = np.empty((self.human_num), dtype=int)
        for l in range(self.human_num):
            for k in range(self.veh_num):
                if self.z_var[l, k].x > 0.5:
                    z_sol[l, k] = 1.0
                    human_in_team[l] = k
        if self.alpha_var is not None:
            alpha_sol = np.zeros(self.veh_num, dtype=np.float64)
            for k in range(self.veh_num):
                alpha_sol[k] = self.alpha_var[k].x
            print('alpha_sol = ', alpha_sol)
        return route_node_list, route_time_list, team_list, y_sol, human_in_team, z_sol
