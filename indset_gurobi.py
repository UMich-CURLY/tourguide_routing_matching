import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB


# Initialize the graph
node_num = 100 # Size of the graph

A = np.random.randint(2, size = (node_num, node_num))
A = np.tril(A) - np.diag(np.diag(A))
A = A + A.T # Finally get the adjacency matrix

G = nx.from_numpy_matrix(A)
print('A = ', A)

# Create the mip solver with the SCIP backend.
solver = gp.Model("MaxIndependentSet")


x = solver.addVars(node_num, vtype=GRB.BINARY, name='x')

obj = 0.0

for i in range(node_num):
    obj += x[i]
solver.setObjective(obj, GRB.MAXIMIZE)


print('Number of variables =', solver.getAttr('NumVars'))

for i in range(node_num):
    for j in range(i+1, node_num):
        if A[i, j] == 1:
            constr_name = 'constr' + str(i) + ',' + str(j)
            solver.addConstr(x[i] + x[j] <= 1.0, constr_name)
print('Number of constraints =', solver.getAttr('NumConstrs'))

solver.optimize()
status = solver.getAttr('Status')	

x_sol = np.zeros(node_num, dtype=int)

if status == GRB.OPTIMAL:
    for i in range(node_num):
        if x[i].x > 0.5:
            x_sol[i] = 1
        print('x[', i, '] =', x[i].x)
else:
    print('The problem does not have an optimal solution.')

time_used = solver.getAttr('Runtime')
print('\nAdvanced usage:')
print('Problem solved in %f seconds' % time_used)
print('Problem solved in %d iterations' % solver.getAttr('IterCount'))
print('Problem solved in %d branch-and-bound nodes' % solver.getAttr('NodeCount'))

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

