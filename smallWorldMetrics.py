
import numpy as np
from scipy import linalg
import itertools
import random
import matplotlib.pyplot as plt
from . import initializeRewireNetworks as rew

# OMPUTES CLUSTERING COEFFICIENT, PATH LENGTH AND SMALL WORLDNESS FOR BINARY AND WEIGHTED NETWORKS


# BINARY NETWORKS

# COMPCLUSTERCOEF calculates the clustering coefficient of a binary network using the formula from the Strogatz paper. Section 7.3.1 in the Networks book (2nd edition) by Newman
# Input
# A: symmetric matrix
# Output
# cCw = average clustering coefficient
def compClusterCoef(A):

    deg = np.sum(A > 0, axis=1)
    # check that there is no one or zero degree vertex
    nodesArray = np.arange(A.shape[0])
    indKeep = np.where(deg > 1)[0]
    if np.array_equal(indKeep, nodesArray) is False:
        deg = deg[indKeep]

    Nv = np.zeros(len(deg))
    for i, k in enumerate(indKeep):  # for each node that is connected more than once
        ind = np.where(A[k, :] > 0)[0]
        combs = list(itertools.combinations(ind, 2))
        # print(combs)
        counter = 0
        for ll in range(len(combs)):
            if A[combs[ll]] == 1:
                # print(combs[ll])
                counter += 1
        Nv[i] = counter

    cCAll = Nv / (0.5 * deg * (deg - 1))
    # print(cCAll)
    cC = np.sum(cCAll) / len(deg)
    return cC  # returns average clustering coefficient


# COMPINVPATHLENGTH computes the average inverse length path of a binary adjacency matrix using breadth first search algorithm. Computing the inverse instead of the path length is convenient for disconnected graph because then an infinite distance becomes 0 when inverting (1/infinite = 0 ). This is also called the harmonic mean distance between nodes, i.e. the average of the inverse distances 'Networks'  M. Newman, p. 172
# INPUT:
# A: adjacency matrix
# OUTPUT:
# avInvPathLength: average inverse path length of each node to each other node
def compInvPathLength(A):

    vertices = A.shape[0]
    denom = (vertices * (vertices - 1) / 2.0)
    deg = np.sum(A, axis=1)
    indZero = np.where(deg == 0)[0]
    if indZero.size > 0:
        #print('There is an adjacency matrix with no connections. We remove this row/column')
        # print(indZero)
        A = np.delete(A, indZero, axis=0)
        A = np.delete(A, indZero, axis=1)

    Adj = makeDictFromMatrix(A)
    invPathLength = 0.0
    for s in range(A.shape[0] - 1):
        level = breadthFirstSearch(s, Adj)
        for k in range(s + 1, A.shape[0]):
            if k in level:
                invPathLength += 1.0 / level[k]
                #print('inv path length between %d and %d is %f' % (s, k, 1.0 / level[k]))
            # else:
                #print('There is a disconnection between %d and %d' % (s, k))

    #print('invPathLength is %f' % invPathLength)
    avInvPathLength = invPathLength / denom

    return avInvPathLength


# MAKEDICTFROMMATRIX takes an adjacency matrix and returns a dictionary with keys 0 to n-1 vertices, each having a list of its neighbors
# will be used for BREADTHFIRSTSEARCH function which in its turn will be used for COMPINVPATHLENGTH
# INPUT:
# A: binary adjacency matrix
# OUTPUT
# Adj: dictionary
def makeDictFromMatrix(A):

    Adj = {}
    for k in range(A.shape[0]):
        ind = np.where(A[k, :] > 0)[0]
        Adj[k] = ind

    return Adj


# BREADTHFIRSTSEARCH measures the distance of one node with the rest of the nodes in the graph using breadth first search algorithm
# will be used for COMPINVPATHLENGTH
# INPUT:
# s: the node number for which we measure its distance with each other node
# Adj: dictionary with the connections of each node
# OUTPUT
# level: dictionary with the distances level[k] = r, r is the distance of node s to node k
def breadthFirstSearch(s, Adj):

    level = {s: 0}
    #parent = {s: 'None'}
    i = 1
    frontier = [s]  # level i-1
    while frontier:
        nextP = []  # level i
        for kk in frontier:
            for jj in Adj[kk]:  # neighbors of node kk
                if jj not in level:
                    level[jj] = i
                    # parent[jj] = kk  # parent is used to traverse the path
                    nextP.append(jj)
        frontier = nextP
        i += 1

    return level


# COMPNORMMETRICS computes the normalized small worldness, clusterning coefficient and path length from a binary adjacency matrix. The formula for small worldness is SNorm = (C/Crand)*(invL/invLrand)
# INPUT:
# A:adjacency matrix
# OUTPUT:
# SNorm: normalized small worldness
# cCNorm: normalized clustering coefficient
# LNorm = normalized path length
def compNormMetrics(A):

    vertices = A.shape[0]
    edges = int(np.sum(A) / 2)

    Arand = rew.generateBinaryRandSymAdj(vertices, edges)

    cCRand = compClusterCoef(Arand)
    invLRand = compInvPathLength(Arand)

    cC = compClusterCoef(A)
    invL = compInvPathLength(A)

    L = 1. / invL
    LRand = 1. / invLRand
    cCNorm = (cC / cCRand)
    LNorm = (L / LRand)
    SNorm = cCNorm / LNorm

    return (SNorm, cCNorm, LNorm)


# WEIGHTED NETWORKS


# COMPWEIGHTCLUSTERCOEF calculates the clustering coefficient of a weighted network using the formula from the paper. Added a 0.5 factor in the denominator to give the same result as the compClusterCoef for the binary case.
# The architecture of complex weighted networks, Barrat et al. PNAS, 2004
# Input
# A: weighted symmetric matrix
# Output
# cCw = average weighted clustering coefficient
def compWeightClusterCoef(A):

    deg = np.sum(A > 0, axis=1)
    strength = np.sum(A, axis=1)  # added weights for each node

    # check that there is no zero or one degree vertex
    nodesArray = np.arange(A.shape[0])
    indKeep = np.where(deg > 1)[0]
    if np.array_equal(indKeep, nodesArray) is False:
        print('There are %d nodes with zero or one degre' % (nodesArray.size - indKeep.size))
        deg = deg[indKeep]
        strength = strength[indKeep]

    wContrAll = np.zeros(len(deg))
    for i, k in enumerate(indKeep):  # for each node
        ind = np.where(A[k, :] > 0)[0]
        combs = list(itertools.combinations(ind, 2))
        wContr = 0.0

        for ll in range(len(combs)):
            if A[combs[ll]] > 0:
                wContr += (A[k, combs[ll][0]] + A[k, combs[ll][1]]) / 2.0

        wContrAll[i] = wContr

    cCAllw = (1 / (0.5 * strength * (deg - 1))) * wContrAll
    # print(cCAllw)
    cCw = np.sum(cCAllw) / len(deg)

    return cCw

# COMPWEIGHTCLUSTERCOEF2 calculates with a different way the clustering coefficient of a weighted network using the formula from the paper.Added a 0.5 factor in the denominator to give the same result as the compClusterCoef for the binary case.
# Intensity and coherence of motifs in weighted complex networks, Onnela et al. Physical Review E, 2005
# Input
# A: weighted symmetric matrix
# Output
# cCw = average weighted clustering coefficient


def compWeightClusterCoef2(A):

    Anorm = A / np.max(A)
    deg = np.sum(Anorm > 0, axis=1)

    # check that there is no zero or one degree vertex
    nodesArray = np.arange(A.shape[0])
    indKeep = np.where(deg > 1)[0]
    if np.array_equal(indKeep, nodesArray) is False:
        print('There are %d nodes with zero or one degre' % (nodesArray.size - indKeep.size))
        deg = deg[indKeep]

    wContrAll = np.zeros(len(deg))
    for i, k in enumerate(indKeep):  # for each node
        ind = np.where(Anorm[k, :] > 0)[0]
        combs = list(itertools.combinations(ind, 2))
        wContr = 0.0

        for ll in range(len(combs)):
            if Anorm[combs[ll]] > 0:
                wContr += (Anorm[k, combs[ll][0]] * Anorm[k, combs[ll][1]] * Anorm[combs[ll]])**(1.0 / 3.0)

        wContrAll[k] = wContr

    cCAllw = (1 / (0.5 * deg * (deg - 1))) * wContrAll
    cCw = np.sum(cCAllw) / len(deg)

    return cCw


# COMPWEIGHTINVPATHLENGTH finds the average inverse path length between all nodes in a weighted (positive weights) symmetric adjacency matrix using Dijkstra algorithm. This is also called the harmonic mean distance between nodes, i.e. the average of the inverse distances 'Networks'  M. Newman, p. 172
# INPUT:
# A: adjacency matrix
# OUTPUT:
# avPathLength: the average path length
def compWeightInvPathLength(A):

    denom = (A.shape[0] * (A.shape[0] - 1) / 2.0)

    deg = np.sum(A > 0, axis=1)
    indZero = np.where(deg == 0)[0]
    if indZero.size > 0:
        print('There is an adjacency matrix with no connections. We remove this row/column')
        A = np.delete(A, indZero, axis=0)
        A = np.delete(A, indZero, axis=1)

    indNonZero = np.where(A > 0)
    AinvWeights = np.zeros(A.shape)
    AinvWeights[indNonZero] = 1.0 / A[indNonZero]  # we take 1 over the weights, the greater the weight the shorter the route

    vertices = A.shape[0]
    invPathLength = 0.0

    for source in range(vertices - 1):
        dist = dijkstra(AinvWeights, source)
        # print(dist[source + 1:])
        invDistances = 1.0 / dist[source + 1:]
        invPathLength += np.sum(invDistances)

    avInvPathLength = invPathLength / denom

    return avInvPathLength


# DIJKSTRA finds the minimum length between a source node and the rest of the nodes. The matrix is weighted with positive weights
# will be used for COMPWEIGHTINVPATHLENGTH
# INPUT:
# A: adjacency matrix
# source: the source node, a number from 0 to n-1, where n is number of nodes
# OUTPUT:
# dist: an np array with the distances from the source. dist[source]=0
def dijkstra(A, source):

    vertices = A.shape[0]

    dist = np.zeros(vertices)
    dist[:] = np.inf
    dist[source] = 0
    queue = np.arange(vertices)

    leastDistNode = source
    queue = np.delete(queue, leastDistNode)
    while queue.size:  # we remove from the queue everytime the min distance node

        for neighbor in np.intersect1d(np.where(A[leastDistNode, :] > 0)[0], queue, assume_unique=True):  # Relaxation
            alt = dist[leastDistNode] + A[leastDistNode, neighbor]
            if alt < dist[neighbor]:
                dist[neighbor] = alt

        indQueue = np.argmin(dist[queue])
        leastDistNode = queue[indQueue]
        queue = np.delete(queue, indQueue)  # remove from queue

    return dist


# COMPWEIGHTNORMMETRICS gets the normalized small worldness, clustering coefficient and path length from a weighted adjacency matrix. The formula for small worldness is S = (C/Crand)*(invL/invLrand)
# We use the inverse path length to then calculate path length so that we could compute disjoint networks. The distance between two nodes from two disjoint clusters is infinite, the inverse is 0.
# INPUT:
# A:adjacency matrix
# weightDistribution: the weight distribution for the random matrix, default to normal
# mu: the mean of the distribution, default to 1
# sig: the sigma (for normal) or the scale (for lognormal) of the distribution, default to 0.25
# OUTPUT:
# SNorm: normalized small worldness
# cCNorm: normalized clustering coefficient
# LNorm = normalized path length
def compWeightNormMetrics(A, weightDistribution='normal', mu=1., sig=0.25):

    vertices = A.shape[0]
    edges = int(np.sum(A > 0) / 2)

    Arand = rew.generateWeightRandSymAdj(vertices, edges, weightDistribution, mu, sig)
    cCRand = compWeightClusterCoef(Arand)

    invLRand = compWeightInvPathLength(Arand)

    cC = compWeightClusterCoef(A)
    invL = compWeightInvPathLength(A)

    L = 1. / invL
    LRand = 1. / invLRand
    cCNorm = (cC / cCRand)
    LNorm = (L / LRand)

    SNorm = cCNorm / LNorm

    return (SNorm, cCNorm, LNorm)
