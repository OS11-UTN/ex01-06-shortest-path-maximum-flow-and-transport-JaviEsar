import numpy as np

# Dijkstra algorithm to find the shortest path in a graph represented as a cost node to node matrix from the
# beginning node startNode to the finish node endNode
def DijkstraAlg (cnn, startNode, endNode):
    predecesor = np.zeros(cnn.shape[0])     # Predecesor array
    done = np.ones(cnn.shape[0])            # Computed node flag array
    cost = np.ones(cnn.shape[0])*np.inf     # Cost to reach each node

    cost[startNode] = 0

    # Find node to expand (minimum cost, assures no other path will later reach the node with a lower cost
    # than the one expanded)
    currNode = np.min(cost).astype(int)


    while currNode != endNode:
        #Mark the node as expanded
        done[currNode] = 0

        # Find the existing arcs for that node
        arcs = np.argwhere(cnn[currNode])

        # For each arc, calculate the cost to reach the next node, and if it is lower than the previous value,
        # replace it and mark the current node as its predecesor
        for available in arcs:
            nextNode = available[0]
            newCost = cnn[currNode, nextNode] + cost[currNode]
            if newCost < cost[nextNode]:
                cost[nextNode] = newCost
                predecesor[nextNode] = currNode

        # Find the next node
        nextCost = np.inf
        for available in np.argwhere(done).astype(int):
            nextNode = available[0]
            if cost[nextNode] < nextCost:
                nextCost = cost[nextNode]
                currNode = nextNode


    # Reconstruct path from predecesor array
    result = list([endNode])
    while currNode != startNode:
        currNode = predecesor[currNode].astype(int)
        result.insert(0, currNode)

    # Return
    return result, cost[endNode]





# Cost to go from node 'row' to node 'column'
cnn = np.array([
        [0, 2, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 5],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0]
    ])


res, cost = DijkstraAlg(cnn, 0, 5)

# Print Result
print("Solver: Dijkstra")
print("Shortest Path:")
for i in res:
    print(i+1, end=" -> ")

print("Objective Function Value: ", cost)
