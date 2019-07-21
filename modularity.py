import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import random

# FINDS CLUSTERS OF NODES, AND EVALUATES THEIR MODULARITY INDEX, THE MORE DENSE THE CONNECTIONS WITHIN CLUSTERS AND THE MORE SPARSE BETWEEN THEM, THE GREATER THE MODULARITY INDEX. THE ALGORITHM IS FROM THE STUDY BELOW BY NEWMAN
# Newman, M. E. (2006). Modularity and community structure in networks. Proceedings of the national academy of sciences, 103(23), 8577-8582.


# MAKEMODULARITYMATRIX takes the adjacency matrix and returns the modularity matrix (Bij = Aij - (d(i)*d(j))/(2m)) that will be used for finding communities within the graph
# INPUT:
# A: the adjacency matrix
# OUTPUT:
# B: the modularity matrix. Similarly to the Laplacian matrix, the sum of its rows or columns should be zero
def makeModularityMatrix(A):

    deg = np.sum(A, axis=1)
    vertices = deg.size
    totalConnections = np.sum(deg) / 2.0

    expConnTemp = np.zeros((vertices, vertices))
    denom = 2 * totalConnections
    for i in range(vertices):
        expConnTemp[i, i] = (deg[i] * deg[i]) / (2 * denom)  # divide by 2 because we are going to add after the iterations
        for j in range(i + 1, vertices):
            expConnTemp[i, j] = (deg[i] * deg[j]) / denom

    expConnMatrix = expConnTemp + np.transpose(expConnTemp)
    B = A - expConnMatrix

    return B, totalConnections


# MAKEMODULARITYMATRIXFROMPARTITION estimates the modularity matrix of a partition or subgraph. It is equation [6] from (Newman,PNAS, 2006). Used for the iterative partitioning
# INPUT:
# B: modularity matrix of the full graph
# partitionInd: indeces of the nodes of the subgraph
# OUTPUT:
# Bpart: the modularity matrix of the partition. Similarly to the modularity matrix of the full graph, the sum of its rows or columns sum to zero
def makeModularityMatrixFromPartition(B, partitionInd):

    Btemp = B.copy()

    # kind of like an outer product to get the right indices, it does not work like matlab (dahhh)
    Bpart = Btemp[np.ix_(partitionInd, partitionInd)]

    for i in np.arange(Bpart.shape[0]):
        Bpart[i, i] -= np.sum(Bpart[i, :])

    return Bpart


# DIV2COM divides the graph or subgraph into two communities and gives the modularity index. The algorithm is described in "Modularity and community structure in networks", M.E.J. Newman, PNAS, 2006
# will be used in PREORDERPARTIONING
# INPUT:
# B: the modularity matrix
# OUTPUT:
# s: vector which can take two values (-1,+1) depending on the community that a node is. For example s[j] = -1 indicates that node j is in community 1, s[k] = 1 that node k is in community 2.
# Q: modularity index; up to a multiplicative constant, the number of edges falling within groups minus the expected number in an equivalent network with edges placed at random

#B, totalConnections = makeModularityMatrix(A)
# or
#Bpart = makeModularityMatrixFromPartition(B, partitionInd)
def div2Com(B, totalConnections):

    # ascending order of eigenvalues
    lambdasAsc, vAsc = np.linalg.eigh(B)
    # reverse them: start from largest going to smallest
    lambdas = lambdasAsc[::-1]
    v = np.flip(vAsc, 1)

    # pick the eigenvector corresponding to the largest eigenvalue
    # v1 = v[:, 0:1]  # it has rank 2 by doing 0:1 instead of just 0
    v1 = v[:, 0]  # this has rank 1 (n,)

    # find the positive and negative elements of the eigenvector
    indPos = np.where(v1 > 0)[0]
    indNeg = np.where(v1 <= 0)[0]

    # makes a dictionary with the two partitions,each containing the indices of the nodes belonging to them
    partitionsInd = {}
    partitionsInd[-1] = indNeg
    partitionsInd[1] = indPos

    # when positive make element of s +1, when negative make it -1
    s = np.zeros((v1.size, 1))

    s[indPos] = 1
    s[indNeg] = -1

    # calculating the modularity index
    Qtemp = (s.T@B@s) / (4 * totalConnections)
    Q = float(np.squeeze(Qtemp))

    # alternatively
    #lambdasRank2 = lambdas[:, np.newaxis]
    # num = np.square((v.T)@s) * lambdasRank2
    #Q = np.sum(num) / (4 * totalConnections)

    return partitionsInd, v1, Q


# The structure from which you make repeated bisection using Newman,PNAS, 2006 paper.
# community is a class representing a cluster of nodes. The variables left and right,
# if the cluster of nodes can be partitioned, point to the partitioned clusters, otherwise
# they do not point anywhere. communityInd is the indices of the nodes in the partition
# Q is the modularity index
class community:
    def __init__(self, communityInd=None):
        self.left = None  # left child
        self.right = None  # right child
        self.communityInd = communityInd
        self.Q = None  # this scalar is positive if the nodes in communityInd can be further split. Add them

# the partitionBinaryTree helps build the tree. It initiates its root (self.root), the indices of all the nodes.
# It also gets the modularity matrix B of the original adjacency matrix, and the total connections of the adjacency matrix. These variables will be used throughout.


class partitionBinaryTree:
    def __init__(self, B, totalConnections):
        communityInd = np.arange(B.shape[0])
        self.root = community(communityInd)  # the indices of the root are 0,1,2....numofVertices
        self.B = B
        self.totalConnections = totalConnections

    # PREORDERPARTIONING iteratively partitions a network of nodes using Newman's method
    # INPUT:
    # startNode: the starting point (partitionBinaryTree.root) of the network containing all the indices of the nodes
    # OUTPUT:
    # Qlist: The list of all the modularity indices from the partitions. Add them all together to get the total modularity index
    # communitiesDict: dictionary with the indices of each community. The keys are positive integers starting from 1 to the number of communities
    def preorderPartitioning(self, startNode, Qlist=[], communitiesDict={}):
        # Root ->Left->Right
        if startNode is not None:
            partB = makeModularityMatrixFromPartition(self.B, startNode.communityInd)
            communitiesInd, v1, startNode.Q = div2Com(partB, self.totalConnections)
            #print('Q is ' ,startNode.Q)
            #print('Community1 size is %d, and community 2 size is %d'%(communitiesInd[-1].size,communitiesInd[1].size))
            if startNode.Q > 0 and communitiesInd[-1].size > 0 and communitiesInd[1].size > 0:
                Qlist.append(startNode.Q)
                startNode.left = community(startNode.communityInd[communitiesInd[-1]])
                startNode.right = community(startNode.communityInd[communitiesInd[1]])
                self.preorderPartitioning(startNode.left, Qlist, communitiesDict)
                self.preorderPartitioning(startNode.right, Qlist, communitiesDict)
            else:
                if not communitiesDict:  # if the dictionary is empty, first timer
                    communitiesDict[1] = startNode.communityInd
                else:
                    maxKey = np.max(list(communitiesDict.keys()))
                    newKey = maxKey + 1
                    communitiesDict[newKey] = startNode.communityInd

        return Qlist, communitiesDict


# GETMODULARITYINDEX uses the functions above to get the modularity index value Q of the adjacency matrix A
# INPUT:
# A: the adjacency matrix
# OUTPUT:
# Q: the modularity index
def getModularityIndex(A):

    B, totalConnections = makeModularityMatrix(A)
    graph = partitionBinaryTree(B, totalConnections)
    Qlist, communitiesDict = graph.preorderPartitioning(graph.root, Qlist=[], communitiesDict={})
    #print('for probability = %f and time = %f '%(p,t))
    # print(Qlist)
    Q = np.sum(Qlist)

    return Q


# FOR VISUALIZATION PURPOSES


# REORDERA2COMMUNITIES reorders A according to the communities taken from the communitiesDict dictionary that contains the indices of each communitiy
# INPUT:
# A: the unordered adjacency matrix
# communitiesDict: dictionary of the indices of the communities
# OUTPUT:
# S: adjacency matrix reordered
def reorderA2Communities(A, communitiesDict):
    sizeClusters = []
    ind = []
    for key in communitiesDict.keys():
        sizeClusters.append(communitiesDict[key].size)
        ind.extend(communitiesDict[key])

    S = A.copy()
    S = S[ind, :]
    S = S[:, ind]
    return S


# GETCOLORPARTITIONEDMATRIX outputs a tensor numVerticesXnumVerticesX3 that when inputed to imshow will show a
# version of the adjacency matrix where nodes belonging to the same community are clustered together. The connections
# between nodes that belong to the same community are colored. Different communities have different colors. Connections
# betwwen nodes are not colored.
# INPUT:
# A: the unsorted adjacency matrix
# communitiesDict: a dictionary with the indices in A of each community, i.e. communitiesDict[1] = the indices for the first cluster
# OUTPUT:
# colMatrix: the numVerticesXnumVerticesX3 tensor
def getColorPartitionedMatrix(A, communitiesDict):

    colors = {}
    colors['blue'] = [0, 0, 255]
    colors['green'] = [0, 128, 0]
    colors['yellow'] = [255, 255, 0]
    colors['red'] = [255, 0, 0]
    colors['purple'] = [128, 0, 128]
    colors['teal'] = [0, 128, 128]
    colors['orange'] = [255, 128, 0]
    colors['maroon'] = [128, 0, 0]
    colors['violet'] = [238, 130, 238]
    colors['turquoise'] = [64, 224, 208]
    colors['tan'] = [210, 180, 140]

    sizeClusters = []
    ind = []
    for key in communitiesDict.keys():
        sizeClusters.append(communitiesDict[key].size)
        ind.extend(communitiesDict[key])

    S = A.copy()
    S = S[ind, :]
    S = S[:, ind]

    indZeros = np.where(S == 0)
    kk = np.ones((1, 1, 3))
    colMatrix = S[:, :, np.newaxis] * kk
    colorKeys = list(colors)  # a list of the keys
    startInd = 0
    for i in np.arange(len(sizeClusters)):
        endInd = np.sum(sizeClusters[:i + 1]) - 1
        #print('for the %d cluster, the color is %s' % (i + 1, colorKeys[i]))
        # print(colors[colorKeys[i]])
        #print('startInd is %d, and endInd is %d' % (startInd, endInd))
        colMatrix[startInd:endInd + 1, startInd:endInd + 1, :] = colMatrix[startInd:endInd + 1, startInd:endInd + 1, :] * colors[colorKeys[i]]
        startInd = endInd + 1

    indRemOnes = np.where(colMatrix[:, :, 0] == 1)  # check for the remaining ones
    colMatrix[indRemOnes[0], indRemOnes[1], :] = 0  # [0,0,0] is black

    colMatrix[indZeros[0], indZeros[1], :] = 255  # get the no connections positions 0s and put 255 -> [255,255,255] is white

    #colMatrix = colMatrix / 255.0

    return colMatrix
