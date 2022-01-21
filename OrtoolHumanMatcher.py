import numpy as np
from ortools.linear_solver import pywraplp
# Reference: https://developers.google.com/optimization/introduction/python

class OrtoolHumanMatcher:
    def __init__(self, human_num, veh_num, max_human_in_team):
        '''
        max_human_in_team: (veh_num, )
        '''
        self.solver = pywraplp.Solver.CreateSolver('GLOP')
        self.objective = self.solver.Objective()
        self.z_var = []
        self.team_constraint = []
        self.human_constraint = []
        self.human_num = human_num
        self.veh_num = veh_num
        self.max_human_in_team = max_human_in_team + 0
        # Initialize variables
        for i_human in range(human_num):
            for i_veh in range(veh_num):
                temp_str = 'z_var[' + str(i_human) + ',' + str(i_veh) + ']'
                temp_x = self.solver.NumVar(0, 1, temp_str)
                self.z_var.append(temp_x)
        # Initialize objective function
        self.objective.SetMinimization()
        for i_human in range(human_num):
            for i_veh in range(veh_num):
                i_z = veh_num * i_human + i_veh
                self.objective.SetCoefficient(self.z_var[i_z], 1)
        # Initialize constraints
        # Each robot should guide n humans, where n is in [1, max_number]
        for i_veh in range(veh_num):
            temp_str = 'veh[' + str(i_veh) + ']'
            temp_constraint = self.solver.Constraint(1, max_human_in_team[i_veh] + 0.0, temp_str) # Human number in a team
            for i_human in range(human_num):
                i_z = veh_num * i_human + i_veh
                temp_constraint.SetCoefficient(self.z_var[i_z], 1)
            self.team_constraint.append(temp_constraint)
        # Each human should be assigned to one robot
        for i_human in range(human_num):
            temp_str = 'human[' + str(i_human) + ']'
            temp_constraint = self.solver.Constraint(1, 1, temp_str)
            for i_veh in range(veh_num):
                i_z = veh_num * i_human + i_veh
                temp_constraint.SetCoefficient(self.z_var[i_z], 1)
            self.human_constraint.append(temp_constraint)

    def optimize(self, human_demand_bool, y_sol):
        '''
        y_sol: (veh_num, place_num)
        scores: (human_num, veh_num)
        '''
        veh_num, place_num = y_sol.shape
        reverse_y_sol = 1.0 - y_sol
        assert veh_num == self.veh_num, 'OrtoolHumanMatcher.optimize: assertation error!'
        prior_offset = 0.0
        # Update the objective function
        scores = np.zeros((self.human_num, self.veh_num), dtype=np.float64)
        for i_human in range(self.human_num):
            for i_veh in range(self.veh_num):
                scores[i_human, i_veh] = (human_demand_bool[i_human] * reverse_y_sol[i_veh]).sum() - prior_offset * i_veh
                i_z = self.veh_num * i_human + i_veh
                self.objective.SetCoefficient(self.z_var[i_z], scores[i_human, i_veh])
        # Optimize
        flag_solver = self.solver.Solve()
        assert flag_solver == self.solver.OPTIMAL or flag_solver == self.solver.FEASIBLE, 'MatchLinProg: infeasible'
        # Get solution
        human_in_team = np.empty((self.human_num), dtype=int)
        z_sol = np.zeros((self.human_num, self.veh_num), dtype=np.float64)
        for i_human in range(self.human_num):
            for i_veh in range(self.veh_num):
                i_z = self.veh_num * i_human + i_veh
                temp_z_sol = self.z_var[i_z].solution_value()
                z_sol[i_human, i_veh] = temp_z_sol
                if temp_z_sol > 0.5:
                    human_in_team[i_human] = i_veh
                    break
        best_match_score = scores.min(axis=1)
        curr_match_score = (scores * z_sol).sum(axis=1)
        total_demand = human_demand_bool.sum()
        dropped_demand = curr_match_score.sum()
        min_dropped_demand = best_match_score.sum()
        demand_result = [total_demand, dropped_demand, min_dropped_demand]
        flag_success = True # Since the problem is not infeasible (it passes the assert), the optimization is completed successfully
        # human_in_team_raw = scores.argmin(axis=1)
        # print('human_in_team_raw = ', human_in_team_raw)
        # print('human_in_team_diff = ', human_in_team - human_in_team_raw)
        # print('best_match_score = ', best_match_score)
        # print('curr_match_score = ', curr_match_score)
        return flag_success, human_in_team, z_sol, demand_result
