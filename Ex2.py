import numpy as np
from scipy.optimize import linprog
from Ex0 import cnn2nac

# Cost to go from node 'row' to node 'column'
cnn = np.array([
        [0, 2, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 5],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0]
    ])

# Net flow for each node (>0 is a source / <0 is a sink)
B = np.array([1, 0, 0, 0, 0, -1])

# Node-Arc Matrix and Cost Matrix computation
A, C, arcs = cnn2nac(cnn)

# Decision variables bounds
bounds = tuple([0, None] for arcs in range(C.shape[0]))




# Solve the linear program using interior-point
res = linprog(C, A_eq=A, b_eq=B, bounds=bounds, method='interior-point')

# Print Result
print("Solver: Interior-Point")
print("Raw Solution: ", res.x)
print("Shortest Path:")
for i in range(res.x.shape[0]):
    if res.x[i] > 1e-3:
        print(arcs[i]+1, end=" -> ")

print("Objective Function Value: ", res.fun)




# Solve the linear program using simplex
res = linprog(C, A_eq=A, b_eq=B, bounds=bounds, method='simplex')

# Print Result
print("Solver: Simplex")
print("Raw Solution: ", res.x)
print("Shortest Path:")
for i in range(res.x.shape[0]):
    if res.x[i] > 0:
        print(arcs[i]+1, end=" -> ")

print("Objective Function Value: ", res.fun)