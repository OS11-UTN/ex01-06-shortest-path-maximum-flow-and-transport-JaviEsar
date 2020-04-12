import numpy as np
import array as arr

# Ford-Fulkerson algorithm to find the shortest path in a graph represented as a upper bound node to node matrix
# from the beginning node startNode to the finish node endNode
def FordFulkersonAlg (ubnn, startNode, endNode):
    maxFlow = 0
    ubnnCopy = ubnn.copy()

    # Until no more paths can be found
    currNode = endNode
    while currNode == endNode:
        # Depth First Search
        prev = arr.array('i', (-1 for i in range(ubnn.shape[0])))    # Previous node
        next = arr.array('i', (-1 for i in range(ubnn.shape[0])))    # Next node
        capacity = np.ones(ubnn.shape[0]) * np.inf                  # Capacity of used arcs (from starting node)
        currNode = startNode                                        # First node
        # Search until endNode is reached (next[endNode] = -1) or no possible paths are left (prev[endNode] = -1)
        while (currNode != -1) & (currNode != endNode):
            # For each possible node
            next[currNode] += 1
            # Check if the arc exists
            if ubnn[currNode, next[currNode]] > 0:
                # Check if the node was not visited already
                if next[next[currNode]] == -1:
                    # Proceed to the next node
                    prev[next[currNode]] = currNode
                    capacity[currNode] = ubnn[currNode, next[currNode]]
                    currNode = next[currNode]
            # Check if there are no arcs left
            if next[currNode] == ubnn.shape[0]-1:
                # Go back to the previous node and continue the search from where it was left
                next[currNode] = -1
                capacity[currNode] = np.inf
                currNode = prev[currNode]
                prev[next[currNode]] = -1

        # If a path was found
        if currNode == endNode:
            # Update max total flow
            maxCapacity = min(capacity)
            maxFlow += maxCapacity

            # Update max capacity of each arc
            currNode = startNode
            while currNode != endNode:
                ubnn[currNode, next[currNode]] -= maxCapacity
                ubnn[next[currNode], currNode] += maxCapacity
                currNode = next[currNode]

    # Find the used arcs
    flow = (ubnnCopy - ubnn)
    arcs = np.argwhere(flow > 0)


    return flow, arcs, maxFlow






# Cost to go from node 'row' to node 'column'
ubnn = np.array([
        [0, 7, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 3],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0]
    ])


flow, arcs, maxFlow = FordFulkersonAlg(ubnn, 0, 5)

# Print Result
print("Solver: Ford-Fulkerson")
print("Transported Units:")
for i in range(arcs.shape[0]):
    print(arcs[i]+1, " -> ", flow[arcs[i, 0], arcs[i, 1]])

print("Objective Function Value: ", maxFlow)
