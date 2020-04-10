import numpy as np
from scipy.optimize import linprog
from Ex0 import cbnn2nacb

# Cost to go from node 'row' to node 'column'
cnn = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0]
    ])

# Capacity upper bound to go from node 'row' to node 'column'
ubnn = np.array([
        [0, 7, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 3],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 2],
        [np.inf, 0, 0, 0, 0, 0]
    ])

# Capacity lower bound to go from node 'row' to node 'column'
lbnn = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

# Net flow for each node (>0 is a source / <0 is a sink)
B = np.array([0, 0, 0, 0, 0, 0])

# Node-Arc Matrix and upper bound computation
A, C, bounds, arcs = cbnn2nacb(cnn, ubnn, lbnn)

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