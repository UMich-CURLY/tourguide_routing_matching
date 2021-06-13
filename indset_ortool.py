import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp

# Initialize the graph
node_num = 100 # Size of the graph

A = np.random.randint(2, size = (node_num, node_num))
A = np.tril(A) - np.diag(np.diag(A))
A = A + A.T # Finally get the adjacency matrix

G = nx.from_numpy_matrix(A)
print('A = ', A)

# Create the mip solver with the SCIP backend.
solver = pywraplp.Solver.CreateSolver('SCIP')
objective = solver.Objective()
objective.SetMaximization()

x = []
for i in range(node_num):
    tempx = solver.IntVar(0.0, 1.0, 'x[' + str(i) + ']')
    x.append(tempx)
    objective.SetCoefficient(tempx, 1)
print('Number of variables =', solver.NumVariables())

for i in range(node_num):
    for j in range(i+1, node_num):
        if A[i, j] == 1:
            solver.Add(x[i] + x[j] <= 1.0)
print('Number of constraints =', solver.NumConstraints())

status = solver.Solve()

x_sol = np.zeros(node_num, dtype=int)

if status == pywraplp.Solver.OPTIMAL:
    for i in range(node_num):
        if x[i].solution_value() > 0.5:
            x_sol[i] = 1
        print('x[', i, '] =', x[i].solution_value())
else:
    print('The problem does not have an optimal solution.')

time_used = solver.wall_time() / 1000
print('\nAdvanced usage:')
print('Problem solved in %f seconds' % time_used)
print('Problem solved in %d iterations' % solver.iterations())
print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

print('max independent set contains', x_sol.sum(), 'nodes')
print('x_sol = ', x_sol)

colours = []
for i in range(node_num):
    if x_sol[i] == 1:
        colours.append('red')
    else:
        colours.append('blue')

plt.figure()
nx.draw_networkx(G, node_color=colours)
plt.show()
