import numpy as np
import gurobipy as gp
from gurobipy import GRB


class GurobiRoutingSolver:
    def __init__(self, veh_num, node_num, human_num, demand_penalty, time_penalty, flag_solver_type):
        self.LARGETIME = 1000.0
        self.veh_num = veh_num
        self.node_num = node_num
        self.human_num = human_num
        # self.demand_penalty = demand_penalty
        self.time_penalty = time_penalty

        self.start_node = self.node_num - 2
        self.end_node = self.node_num - 1

        self.solver = gp.Model("Routing")
        self.x_var = self.solver.addVars(self.veh_num, self.node_num, self.node_num, vtype=GRB.BINARY, name='x')
        self.y_var = self.solver.addVars(self.veh_num, self.node_num-2, vtype=GRB.BINARY, name='y')
        self.time_var = self.solver.addVars(self.veh_num, self.node_num, vtype=GRB.CONTINUOUS, name='q', lb=0.0, ub=self.LARGETIME)
        self.max_time_var = self.solver.addVar(0.0, self.LARGETIME, 1.0, GRB.CONTINUOUS, "q_max")

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
        return result_dict

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
            for i in range(self.node_num-2):
                # Variable Relationship Constraint: Incoming edges == y
                constr = 0
                for j in range(self.node_num):
                    constr += self.x_var[k,j,i]
                constr_name = 'task[' + str(k) + ',' + str(i) + ']'
                self.solver.addConstr(constr == self.y_var[k,i], constr_name)
                # Flow Constraint: Incoming edge <= outgoing edge from start (redundant but useful for solver)
                constr_name1 = 'task<=start[' + str(k) + ',' + str(i) + ']'
                self.solver.addConstr(constr <= constr1, constr_name1)
        
        # Time constraint
        for k in range(self.veh_num):
            for i in range(self.node_num):
                for j in range(self.node_num):
                    constr = self.time_var[k,i] - self.time_var[k,j] + edge_time[k,i,j] + node_time[k,i] - self.LARGETIME * (1 - self.x_var[k,i,j])
                    constr_name = 'edge_time[' + str(k) + ',' + str(i) + ',' + str(j) + ']'
                    self.solver.addConstr(constr <= 0, constr_name)
        for k in range(self.veh_num):
            for i in range(self.node_num):
                constr_name = 'max_time[' + str(k) + ',' + str(i) + ']'
                self.solver.addConstr(self.time_var[k,i] <= self.max_time_var, constr_name)


        # Energy constraint
        # Placeholder

    def get_plan(self):
        route_list = []
        for k in range(self.veh_num):
            curr_node = self.start_node
            route = [curr_node]
            for ii in range(100):
                next_node = -1
                for i in range(self.node_num):
                    if self.x_var[k,curr_node,i].x > 0.5:
                        next_node = i
                        break
                if next_node == -1:
                    break
                route.append(next_node)
                curr_node = next_node
            route_list.append(route)

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
        
        time_mat = np.zeros((self.veh_num, self.node_num), dtype=np.float64)
        for k in range(self.veh_num):
            for i in range(self.node_num):
                time_mat[k,i] = self.time_var[k,i].x
        # print(time_mat)
        return route_list, team_list

