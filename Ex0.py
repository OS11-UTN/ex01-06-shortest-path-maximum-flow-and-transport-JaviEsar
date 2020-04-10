import numpy as np

# Converts a Node-Node matrix to the corresponding Node-Arc matrix
def nn2na (nn):
    # Find the existing arcs
    arcs = np.argwhere(nn)

    # Create the node-arc matrix
    na = np.zeros([nn.shape[0], arcs.shape[0]]).astype(int)

    # For each arc, update the two corresponding entries in the node-arc matrix
    for i in range(arcs.shape[0]):
        na[arcs[i, 0], i] = 1
        na[arcs[i, 1], i] = -1

    # Return
    return na, arcs


# Converts a Cost Node-Node matrix to the corresponding Node-Arc and Cost matrix
def cnn2nac (cnn):
    # Find the existing arcs
    arcs = np.argwhere(cnn)

    # Create the node-arc matrix and cost matrix
    na = np.zeros([cnn.shape[0], arcs.shape[0]]).astype(int)
    c = np.zeros([arcs.shape[0]])

    # For each arc, update the two corresponding entries in the node-arc matrix
    for i in range(arcs.shape[0]):
        na[arcs[i, 0], i] = 1
        na[arcs[i, 1], i] = -1
        c[i] = cnn[arcs[i, 0], arcs[i, 1]]

    # Return
    return na, c, arcs



# Converts a Cost Node-Node matrix, a Upper Bound Node-Node matrix and a Lower Bound Node-Node matrix
# to the corresponding Node-Arc matrix, Cost matrix and bounds tuple
# NOTE: All input matrices must be of the same shape
def cbnn2nacb (cnn, ubnn, lbnn):
    # Verify input
    if (cnn.shape[0] != ubnn.shape[0]) | (cnn.shape[0] != lbnn.shape[0]):
        print("ERROR: Matrix must have the same number of rows")
        return [], [], (), []
    if (cnn.shape[1] != ubnn.shape[1]) | (cnn.shape[1] != lbnn.shape[1]):
        print("ERROR: Matrix must have the same number of columns")
        return [], [], (), []

    # Find the existing arcs
    arcs = np.argwhere((cnn != 0) | (ubnn != 0) | (lbnn != 0))

    # Create the node-arc matrix and bound tuple
    na = np.zeros([ubnn.shape[0], arcs.shape[0]]).astype(int)
    c = np.zeros([arcs.shape[0]])
    ub = []
    lb = []

    # For each arc, update the two corresponding entries in the node-arc matrix
    for i in range(arcs.shape[0]):
        na[arcs[i, 0], i] = 1
        na[arcs[i, 1], i] = -1
        c[i] = cnn[arcs[i, 0], arcs[i, 1]]
        bound = lbnn[arcs[i, 0], arcs[i, 1]]
        if bound == np.inf:
            bound = None
        lb.append(bound)
        bound = ubnn[arcs[i, 0], arcs[i, 1]]
        if bound == np.inf:
            bound = None
        ub.append(bound)

    # Return
    bounds = tuple(zip(lb, ub))
    return na, c, bounds, arcs



def test ():
    var1 = np.array([[0,1,1],[0,0,0],[0,0,0]])
    var2, var3 = nn2na(var1)
    print("Arcs:\n", var3+1, "\nNode-Arc Matrix:\n", var2)

    var1 = np.array([[0,2,1],[0,0,0],[0,0,0]])
    var2, var3, var4 = cnn2nac(var1)
    print("Arcs:\n", var4+1, "\nNode-Arc Matrix:\n", var2, "\nCost-Arc Matrix:\n", var3)


#test()
