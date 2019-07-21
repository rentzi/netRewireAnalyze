import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import random
from datetime import datetime as dt

# CREATE RANDOM OR REGULAR ADJACENCY MATRICES, REWIRE ACCORDING TO HEAT DIFFUSION, COUPLED MAPS, WATTS-STROGATS ALGORITHMS


# INITIALIZE ADJACENCY MATRICES - BINARY OR WEIGHTED RANDOM, REGULAR NETWORKS

# GENERATEBINARYRANDSYMADJ generates a binary random symmetric adjacency matrix
# INPUT
# vertices: number of vertices or nodes
# edges: number of edges
# OUTPUT
# A: random symmetric verticesXvertices adjacency matrix with 0s at Aij Aji if the i-j are not connected, 1 if they are
def generateBinaryRandSymAdj(vertices, edges):

    maxConnections = int(vertices * (vertices - 1) / 2)  # the maximum connections a network can have

    if edges > maxConnections or edges < 0:
        print('The number of edges are not within the permitted range')
        return -1

    # Get the indices of 1s of a matrix with 1s only on its upper triangular part
    upperTriangOnes = np.triu(np.ones((vertices, vertices)) - np.eye(vertices))
    ind = np.where(upperTriangOnes)
    # Get a random sample of those indices (#edges)
    xxRand = np.random.permutation(maxConnections)
    indRand = (ind[0][xxRand[:edges]], ind[1][xxRand[:edges]])
    # construct with those indices the upper triangular part of the adjacency matrix
    aTemp = np.zeros((vertices, vertices))
    aTemp[indRand] = 1
    # add the transpose of that matrix to get the lower part and make the matrix symmetric
    A = aTemp + np.transpose(aTemp)

    return A


# GENERATEWEIGHTRANDSYMADJ generates a weighted random symmetric adjacency matrix
# INPUT
# vertices: number of vertices or nodes
# edges: number of edges
# weightDistribution: it can either be normal or lognormal, default to normal
# mu: the mean of the distribution, default to 1
# sig: the sigma (for normal) or the scale (for lognormal) of the distribution, default to 0.25
# OUTPUT
# A: random symmetric verticesXvertices adjacency matrix with 0s at Aij Aji if the i-j are not connected, 0<w<1 if they are. The weights follow a distribution specified
def generateWeightRandSymAdj(vertices, edges, weightDistribution='normal', mu=1., sig=0.25):

    maxConnections = int(vertices * (vertices - 1) / 2)  # the maximum connections a network can have
    epsilon = 0.05

    if edges > maxConnections or edges < 0:
        print('The number of edges are not within the permitted range')
        return -1

    # I use lognormal for the time being. We can make it into a parameter
    if weightDistribution == 'lognormal':
        # mu, sig = 0., 1.
        randWeights = np.random.lognormal(mean=mu, sigma=sig, size=edges)
    elif weightDistribution == 'normal':
        #mu, sig = 1., 0.25
        randWeights = np.random.normal(loc=mu, scale=sig, size=edges)
        ind = np.where(randWeights < 0)
        randWeights[ind] = epsilon

    normRandWeights = randWeights / np.max(randWeights)  # normalize so that the values are between 0 and 1

    # Get the indices of 1s of a matrix with 1s only on its upper triangular part
    upperTriangOnes = np.triu(np.ones((vertices, vertices)) - np.eye(vertices))
    ind = np.where(upperTriangOnes)
    # Get a random sample of those indices (#edges)
    xxRand = np.random.permutation(maxConnections)
    indRand = (ind[0][xxRand[:edges]], ind[1][xxRand[:edges]])
    # construct with those indices the upper triangular part of the adjacency matrix
    aTemp = np.zeros((vertices, vertices))
    aTemp[indRand] = normRandWeights
    # add the transpose of that matrix to get the lower part and make the matrix symmetric
    A = aTemp + np.transpose(aTemp)

    return A


# CREATEREGULARADJMATRIX creates an adjacency matrix that is regular, connected to its neighboring indeces in the way explained in the seminal paper by
# Collective Dynamics of Small World Networks, Watts & Strogatz, Nature, 1998
# INPUT:
# vertices: the number of vertices of the network
# neighbors: the number of connected neighbors from each side of the node
# weightDistribution: the weight distribution used, either 'binary', 'normal' or 'lognormal'
# OUTPUT:
# A: the regular adjacency matrix
def createRegularAdjMatrix(vertices, neighbors, weightDistribution):

    edges = 2 * neighbors * vertices
    if weightDistribution == 'binary':
        weights = np.ones(edges)
    elif weightDistribution == 'lognormal':
        mu, sig = 0., 1.  # mean and standard deviation
        weights = np.random.lognormal(mean=mu, sigma=sig, size=edges)
    elif weightDistribution == 'normal':
        mu, sig = 1., 0.25
        weights = np.random.normal(loc=mu, scale=sig, size=edges)

    neighborsInd = np.concatenate((np.arange(vertices - neighbors, vertices), np.arange(vertices), np.arange(neighbors)), axis=0)

    A = np.zeros((vertices, vertices))

    for vert in np.arange(vertices):
        nodes2Connect = neighborsInd[vert:vert + 2 * neighbors + 1]
        A[vert, nodes2Connect[-neighbors:]] = weights[2 * vert]
        A[vert, nodes2Connect[:neighbors]] = weights[2 * vert + 1]

    return A

#####################################################################################

# REORDER THE NODES IN THE ADJACENCY MATRIX ACCORDING TO THE MAGNITUDE OF THE EIGENVALUES OF ITS ADJACENCY MATRIX AND SHOW THEM


# REORDERA2VISUALIZE reorders the rows/columns of A so that we can see the clustering.
# It takes the eigenvector corresponding to the 2nd smallest eigenvalue of the laplacian->L = D-A and reorders it. Uses those indices to reorder the A matrix
# will be used for PLOTADJMATRIX
# INPUT:
# SS: the initial unordered matrix
# OUTPUT:
# A: the SS matrix ordered
def reorderA2Visualize(SS):

    A = SS.copy()
    deg = np.sum(A, axis=1)
    ##################################
    # if there is a degree 0, in the inversion it stays 0
    vertices = A.shape[0]
    I = np.eye(vertices)
    indDeg = np.where(deg > 0)[0]
    deg2 = np.zeros(deg.size)
    deg2[indDeg] = 1.0 / np.sqrt(deg[indDeg])

    deginv = np.expand_dims(deg2, axis=1)
    L = I - ((A * deginv) * np.transpose(deginv))  # Get the normalized Laplacian

    # decompose the matrix to its eigenvectors/eigenvalues
    eigval, eigvec = np.linalg.eigh(L)
    ##################################################

    eigSortInd = np.argsort(eigval)
    # takes second eigenvalue. The first is trivial solution. Next way to take more than one eigenvectors and do clustering
    clusterInd = np.argsort(eigvec[:, eigSortInd[1]])
    A = A[clusterInd, :]
    A = A[:, clusterInd]

    return A


# REWIRING

# REWIRING BASED ON HEAT KERNEL
# Jarman, N., Steur, E., Trengove, C., Tyukin, I. Y., & Van Leeuwen, C. (2017). Self-organisation of small-world networks by adaptive rewiring in response to graph diffusion. Scientific reports, 7(1), 13158.
# and the current work


# REWIREHEATKERNEL rewires iteratively a matrix A. At each iteration the rewiring can be random (probability= pRandRewire) or according to a heat dispersion function (probability = 1-pRandRewire). Works for both binary and weighted initial networks since this implementation just redistributes the weights
# INPUT
# Arand: random symmetric adjacency matrix
# pRandRewire: probability of random rewiring
# rewirings: number of iterations the wiring take place
# tau: heat dispersion parameter
# eigenInd: the indices of the decomposed in eigenvalue/eigenvector laplacian for which we calculate the heat equation. The eigenvalues are in ascending order, we can select which we want to calculate the heat equation. By default the whole laplacian is selected
# OUTPUT
# A: returns a rewired symmetric matrix
def rewireHeatKernel(Arand, pRandRewire, rewirings, tau, eigenInd=[]):

    A = Arand.copy()

    vertices = A.shape[0]
    I = 1.0 * np.eye(vertices)

    for k in range(rewirings):

        deg = np.sum(A > 0, axis=1, keepdims=False)  # deg[i] = the number of connections of i+1 node
        vNonZeroInd = np.where((deg > 0) & (deg < vertices - 1))  # take the indices of the nodes with nonzero but not numofVertices degree
        if len(vNonZeroInd[0]) == 0:
            print('For tau = %f, and p(rand) = %f, we have graph with either fully connected or nonconnected nodes' % (tau, pRandRewire))
            return A

        vRandInd = np.random.choice(vNonZeroInd[0])  # pick one of those indices at random

        indAll = np.arange(vertices)  # 0:vertices-1
        indMinusV = np.delete(indAll, vRandInd)  # remove the vRandInd index, ie for VRandInd=2 indMinusV = 0,1,3,..
        ANotVCol = 1.0 * np.logical_not(A[indMinusV, vRandInd])  # take the actual vector and make inversions 0->1 and 1->0
        if np.random.random_sample() >= pRandRewire:  # rewire by network diffusion

            L = getNormLaplacian(A)

            if len(eigenInd) == 0:
                h = linalg.expm(-tau * L)  # heat dispersion component
            else:
                h = getHeatEigenDecomp(L, tau, eigenInd)

            indTestable = np.where(A[:, vRandInd] > 0)[0]  # do not include the 0s
            u1Testable = np.argmin(h[indTestable, vRandInd])  # check the heat kernel minimum value of the nodes connected to vRandInd
            u1 = indTestable[u1Testable]

            indANotVCol = np.where(ANotVCol > 0)[0]
            indNotConnected = indMinusV[indANotVCol]
            u2IndTemp = np.argmax(h[indNotConnected, vRandInd])  # what would happen if the ones that were connected to vRandInd were connected to it and was applied to them the heat kernel. Get the ind of the maximum from those nodes. this will be used for reconnection

            u2 = indNotConnected[u2IndTemp]  # get the right u2 node
        else:  # now we just randomly rewire
            noConnIndex = np.argwhere(ANotVCol)
            u2IndTemp = noConnIndex[np.random.choice(noConnIndex.size)][0]
            u2 = indMinusV[u2IndTemp]  # pick randomly a nonconnection to vRandInd node

            AOnesInd = np.argwhere(A[:, vRandInd] > 0)
            u1 = AOnesInd[np.random.choice(AOnesInd.size)][0]  # pick randomly a connected node to vRandInd

        A[u2, vRandInd] = A[u1, vRandInd]
        A[vRandInd, u2] = A[u1, vRandInd]
        A[u1, vRandInd] = 0
        A[vRandInd, u1] = 0

    return A


# GETNORMLAPLACIAN gets the normalized laplacian matrix LNorm from a symmetric adjacency matrix A
# INPUT
# A: a NxN symmetric adjacency matrix from which we get the normalized Laplacian
# OUTPUT
# LNorm: the NxN normalized Laplacian matrix
def getNormLaplacian(A):

    vertices = A.shape[0]
    I = np.eye(vertices)
    deg = np.sum(A, axis=1)
    # if there is a degree 0, in the inversion it stays 0
    indDeg = np.where(deg > 0)[0]
    deg2 = np.zeros(deg.size)
    deg2[indDeg] = 1.0 / np.sqrt(deg[indDeg])

    deginv = np.diag(deg2)
    LNorm = I - deginv@A@deginv

    return LNorm


# GETHEATEIGENDECOMP takes the L, decomposes it into its eigenvectors/values and then selects from the eigenInd the eigenvectors/values from which it will equate the heat equation h. The eigenvalues are in ascending order
# INPUT:
# L: the laplacian matrix
# tau: the time parameter for the heat equation
# eigenInd: the indices that indicate the eigendecomposition
# OUTPUT:
# h: heat equation
def getHeatEigenDecomp(L, tau, eigenInd):

    lambdasAll, vAll = np.linalg.eigh(L)

    lambdas = lambdasAll[eigenInd]
    v = vAll[:, eigenInd]
    vT = np.transpose(v)
    # makes a diagonal matrix from the vector
    lambdasD = np.diag(lambdas)

    hEigenval = linalg.expm(-tau * lambdasD)
    h = v@hEigenval@vT

    return h


# REWIRING BASED ON WATTS AND STROGATS ALGORITHM
# Collective Dynamics of Small World Networks, Watts & Strogatz, Nature, 1998


# REWIRESWN rewires a regular network in the way explained in the seminal paper by Watts and Strogatz
# Collective Dynamics of Small World Networks, Watts & Strogatz, Nature, 1998
# Input:
# Aregular: the network in the form of an adjacency matrix to be rewired
# p: probability of rewiring (between 0 and 1)
# Output:
# A: the rewired adjacency matrix
def rewireSWN(Aregular, p):

    A = Aregular.copy()
    random.seed(dt.now())

    vertices = A.shape[0]
    indAll = np.arange(vertices)  # 0:vertices-1

    for row in np.arange(vertices - 1):
        indMinusV = np.delete(indAll, row)
        # print(indMinusV)
        nonZeroInd = np.where(A[row, :] > 0)[0]
        nonZeroIndNested = np.where(nonZeroInd > row)[0]
        nonZeroIndRow = nonZeroInd[nonZeroIndNested]
        # print(nonZeroIndRow)
        for col in nonZeroIndRow:
            # print(A[row,col])
            if np.random.random_sample() <= p:  # if this valid we attempt the rewiring
                vRandInd = np.random.choice(indMinusV)
                if A[row, vRandInd] == 0:
                    A[row, vRandInd] = A[row, col]
                    A[vRandInd, row] = A[row, col]
                    A[row, col] = 0
                    A[col, row] = 0

    return A


# REWIRESWNVariation rewires a regular network in the way explained in the seminal paper by Watts and Strogatz with the variation that it picks every time a random edge and performs it for a specified number of times
# Collective Dynamics of Small World Networks, Watts & Strogatz, Nature, 1998
# Input:
# Aregular: the network in the form of an adjacency matrix to be rewired
# p: probability of rewiring (between 0 and 1)
# Output:
# A: the rewired adjacency matrix
def rewireSWNVariation(Aregular, p, iterations):
    A = Aregular.copy()
    random.seed(dt.now())

    vertices = A.shape[0]
    indAll = np.arange(vertices)  # 0:vertices-1

    for iteration in np.arange(iterations):

        flagEdge = 0
        while flagEdge == 0:
            i = random.randint(0, vertices - 1)
            j = random.randint(0, vertices - 1)
            if A[i, j] > 0:
                flagEdge = 1

        indMinusV = np.delete(indAll, i)
        # print(indMinusV)
        if np.random.random_sample() <= p:  # if this valid we attempt the rewiring
            vRandInd = np.random.choice(indMinusV)
            if A[i, vRandInd] == 0:
                A[i, vRandInd] = A[i, j]
                A[vRandInd, i] = A[i, j]
                A[i, j] = 0
                A[j, i] = 0

    return A


# REWIRING BASED ON COUPLED MAPS
# Gong, P., & van Leeuwen, C. (2004). Evolution to a small-world network with chaotic units. EPL (Europhysics Letters), 67(2), 328.


# REWIRECOUPLEDMAP rewires the adjacency matrix using coupled logistic maps according to (Gong, Leeuwen 2004 Europhysics)
# INPUT
# Arand: random symmetric adjacency matrix
# transientTime: the time(iterations) for which you let the states evolve but you do not start the rewiring
# iterations: number of iterations the wiring take place according to the similarity rule
# couplingStrength: the epsilon parameter in the equation
# alpha: the alpha parameter in the logistic map equation
# OUTPUT
# A: returns a rewired symmetric matrix
def rewireCoupledMap(Arand, transientTime, iterations, couplingStrength, alpha):

    A = Arand.copy()

    vertices = A.shape[0]

    stateNodeVec = np.random.uniform(low=-1.0, high=1.0, size=vertices)  # initialize the state of the nodes
    # you iterate for some time (transient time) the states
    for r in range(transientTime):
        stateNodeVec = estimateNodeNextState(A, stateNodeVec, couplingStrength, alpha)

    indAll = np.arange(vertices)
    # You will have in a separate loop here the iterations of the map
    for k in range(iterations):
        deg = np.sum(A, axis=1)  # deg[i] = the number of connections of i+1 node
        if any(deg == 0):
            print('For %d iterations, coupling strength = %f, and bifurcation parameter = %f one of the nodes has no connections. Redoing the rewiring' % (iterations, couplingStrength, alpha))
            return rewireCoupledMap(Arand, transientTime, iterations, couplingStrength, alpha)

        stateNodeVec = estimateNodeNextState(A, stateNodeVec, couplingStrength, alpha)

        nodeRandInd = np.random.choice(vertices)  # pick a random node's ind
        distAllNodes = np.abs(stateNodeVec - stateNodeVec[nodeRandInd])  # measure the absolute distance between node state values

        indMinusSelf = np.delete(indAll, nodeRandInd)  # get the indices of the nodes it is connected to
        minTemp = np.argmin(distAllNodes[indMinusSelf])
        minInd = indMinusSelf[minTemp]

        # We want to find the maximum difference among connected nodes
        AOnesInd = np.argwhere(A[:, nodeRandInd] > 0)
        maxTemp = np.argmax(distAllNodes[AOnesInd])
        maxInd = AOnesInd[maxTemp][0]

        if A[nodeRandInd, minInd] == 0:
            #print('rewiring in %d iteration' % (k))
            A[nodeRandInd, minInd] = A[nodeRandInd, maxInd]
            A[minInd, nodeRandInd] = A[nodeRandInd, maxInd]
            A[nodeRandInd, maxInd] = 0
            A[maxInd, nodeRandInd] = 0

    return A


# ESTIMATENODENEXTSTATE outputs a vector with the t+1 states of the nodes using coupled quadratic logistic maps (Gong, Leeuwen 2004 Europhysics)
# INPUT
# A: adjacency matrix, can be either binary or weighted
# stateNodeVec: the state of nodes at time t
# couplingStrength, alpha: parameters in the equation
# OUTPUT
# nextStateNodeVec: the vector with the t+1 states of the nodes
def estimateNodeNextState(A, stateNodeVec, couplingStrength, alpha):

    deg = np.sum(A, axis=1)
    fstateNodeVec = 1 - alpha * (stateNodeVec**2)
    weightedSumF = np.sum(A * fstateNodeVec, axis=1)  # becomes a row vector and tiled along the rows
    nextStateNodeVec = ((1 - couplingStrength) * fstateNodeVec) + (couplingStrength / deg) * weightedSumF

    return nextStateNodeVec
