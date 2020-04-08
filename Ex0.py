import numpy as np

def nn2na (nn):
    # Find the existing arcs
    arcs = np.argwhere(nn)

    # Create the node-arc matrix
    na = np.zeros([nn.shape[0], arcs.shape[0]]).astype(int)

    # For each arc, update the two corresponding entries in te node-arc matrix
    for i in range(arcs.shape[0]):
        na[arcs[i, 0], i] = 1
        na[arcs[i, 1], i] = -1

    # Return
    return na, arcs



def cnn2nac (cnn):
    # Find the existing arcs
    arcs = np.argwhere(cnn)

    # Create the node-arc matrix and cost matrix
    na = np.zeros([cnn.shape[0], arcs.shape[0]]).astype(int)
    c = np.zeros([arcs.shape[0]])

    # For each arc, update the two corresponding entries in te node-arc matrix
    for i in range(arcs.shape[0]):
        na[arcs[i, 0], i] = 1
        na[arcs[i, 1], i] = -1
        c[i] = cnn[arcs[i, 0], arcs[i, 1]]

    # Return
    return na, c, arcs



def test ():
    var1 = np.array([[0,1,1],[0,0,0],[0,0,0]])
    var2, var3 = nn2na(var1)
    print("Arcs:\n", var3+1, "\nNode-Arc Matrix:\n", var2)

    var1 = np.array([[0,2,1],[0,0,0],[0,0,0]])
    var2, var3, var4 = cnn2nac(var1)
    print("Arcs:\n", var4+1, "\nNode-Arc Matrix:\n", var2, "\nCost-Arc Matrix:\n", var3)


#test()