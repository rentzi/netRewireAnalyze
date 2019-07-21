import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import random

# CALCULATES THE ASSORTATIVITY AND RICH CLUB COEFFICIENTS


# THE TWO FUNCTIONS BELOW CALCULATE THE ASSORTATIVITY COEFFICIENT
# GENERALIZES TO BINARY AND WEIGHTED ADJACENCY MATRICES
# BINARY
# Newman, M. E. (2003). Mixing patterns in networks. Physical Review E, 67(2), 026126.
# Newman, M. (2010). Networks: an introduction. Oxford university press.
# WEIGHTED
# Farine, D. R. (2014). Measuring phenotypic assortment in animal social networks: weighted associations are more robust than binary edges. Animal Behaviour, 89, 141-153.
# Leung, C. C., & Chau, H. F. (2007). Weighted assortative and disassortative networks model. Physica A: Statistical Mechanics and its Applications, 378(2), 591-602.

# GETASSORTCOEF gets the assortativity coefficient for both binary and weighted adj matrices
# INPUT:
# A: the adjacency matrix
# binarizedFlag: if True we do not take the strength of the connections into consideration, defaults to True
# OUTPUT:
# assortCoef: the assortativity coefficient
def getAssortCoef(A, binarizedFlag=True):

    if binarizedFlag == True:
        A = 1.0 * (A > 0)

    B, BNorm = makeModAndNormMod(A)

    strength = np.sum(A > 0, axis=1)  # this is actually degree

    strengthTemp = np.repeat(strength[..., np.newaxis], len(strength), axis=1)
    strengthMatrix = strengthTemp * np.transpose(strengthTemp)

    numMatrix = B * strengthMatrix
    denomMatrix = BNorm * strengthMatrix

    num = np.sum(np.sum(numMatrix))
    denom = np.sum(np.sum(denomMatrix))

    assortCoef = num / denom

    return assortCoef

# MAKEMODANDNORMMOD takes the adjacency matrix and returns the modularity matrix  (Bij = Aij - (str(i)*str(j))/(2m))
# and its normalizing version (Bij = str(i)*delta(i,j) - (str(i)*str(j))/(2m)). will be used for the calculation of the assortativity coeff
# INPUT:
# A: the adjacency matrix
# OUTPUT:
# B: the modularity matrix
# BNorm: the normalizing modularity matrix


def makeModAndNormMod(A):

    strength = np.sum(A, axis=1)
    vertices = strength.size
    totalConnections = np.sum(strength) / 2.0

    expConnTemp = np.zeros((vertices, vertices))
    strengthDiag = np.diag(strength)
    denom = 2 * totalConnections
    for i in range(vertices):
        expConnTemp[i, i] = (strength[i] * strength[i]) / (2 * denom)  # divide by 2 because we are going to add after the iterations
        for j in range(i + 1, vertices):
            expConnTemp[i, j] = (strength[i] * strength[j]) / denom

    expConnMatrix = expConnTemp + np.transpose(expConnTemp)

    B = A - expConnMatrix
    BNorm = strengthDiag - expConnMatrix

    return B, BNorm


# CALCULATE RICH CLUB COEFFICIENTS

# GETRICHCLUBCOEF gets the unormalized rich club coefficient.
# In the binary case the numerator is the number of connections between rich club nodes over the maximum possible number of connections
# between rich club nodes. Thus in binary we count the num of connections in the denominator as if rich club nodes are fully connected.
# In the weighted case the numerator is the total strength of connections between rich club nodes over the strength for the same number
# of connections but for the largest weights. So it is the maximum possible strength given the connectivity. Thus, in weighted we
# preserve topology but change the weights
# INPUT:
# A: the adjacency matrix
# k: a scalar positive integer value indicating that the nodes consisting the rich club should have degree greater than k
# binaryFlag: if True the adjacency matrix A should be binary, if not it is weighted. The two different ways rich club is calculated are explained above
# OUTPUT:
# richClub: a tuple that is different for the 'binaryFlag = True' and 'binary Flag = False' cases. More specifically:
# if binaryFlag = True, then richClub = (phiK, mK, nK), where phiK = the rich club coef value; mK = the number of connections between rich club nodes and; nK = the number of rich club nodes
# if binaryFlag = False then richClub = (phiK, mK, nK, totalStr, maxStr) , where the first 3 variables in the tuple have the same meaning as in the True case, and the last two are: totalStr = the total strength of the connections between rich club nodes and; maxStr = the maximum possible strength between rich club nodes for the same number of connections
def getRichClubCoef(A, k, binaryFlag=True):

    deg = np.sum(A > 0, axis=0)

    indK = np.where(deg > k)[0]
    if indK.size == 0:
        if binaryFlag == True:
            return np.nan, 0, 0
        else:
            return np.nan, 0, 0, 0, 0

    AKTemp = A[indK, :]
    AK = AKTemp[:, indK]

    mK = np.sum(np.sum(AK > 0)) / 2.0
    nK = AK.shape[0]
    if binaryFlag == True:
        phiK = 2.0 * mK / (nK * (nK - 1))
        richClub = (phiK, mK, nK)
    else:
        totalStr = np.sum(np.sum(AK)) / 2.0
        AFlat = A.flatten()
        AFlat[::-1].sort()
        maxStr = np.sum(AFlat[:int(mK)])
        phiK = totalStr / maxStr
        richClub = (phiK, mK, nK, totalStr, maxStr)

    return richClub

# RANDOMIZEKEEPDEGDISTR gets the random network that will be used for the calculation of the normalized rich club coeff. For binary networks
# We construct the random network by performing a number of double edge swaps from the original network. A double edge swap involves randomly
# choosing two connections a-b and c-d, and replacing them with the connections a-c and b-d. If the later connections already exist, then the
# algorithm does not count that iteration and chooses again two random connections. The resulting random network has the same degree distribution
# as the input adj matrix.
# Towlson, Emma K., et al. "The rich club of the C. elegans neuronal connectome." Journal of Neuroscience 33.15 (2013): 6380-6387.
# INPUT:
# Adj: the adjacency matrix
# swaps = the number of double edge swaps, you do this for binary or binarized networks. the randomized control has the
# same degree sequence as the original adjacency matrix
# OUTPUT:
# A: the random network
# rewire: the number of rewirings, if everything goes well it should be the same number as the one in variable swaps


def randomizeKeepDegDistr(Adj, swaps):

    threshConsecFail = 200
    A = Adj.copy()

    flagStay = True
    failAccum = 0
    rewire = 0
    while flagStay == True:

        deg = np.sum(A > 0, axis=0)

        vNonZeroInd = np.where((deg > 0) & (deg < A.shape[0] - 1))[0]
        if len(vNonZeroInd) == 0:
            print('we have graph with either fully connected or nonconnected nodes')
            flagStay = False
        else:
            # i---j
            # X   X
            # k---d
            i = np.random.choice(vNonZeroInd)  # selects the value not the index from vNonZeroInd  #i
            jCand = np.where(A[vNonZeroInd, i] > 0)[0]  # j
            if jCand.size == 0:
                failAccum += 1
                #print('Failed to find j, failAccum is %d'%failAccum)
            else:
                j = np.random.choice(vNonZeroInd[jCand])

                # Finding k
                indMinusIJ = np.setdiff1d(vNonZeroInd, [i, j])  # remove the i and j elements from vNonZeroInd
                indNotConnected = np.where(A[indMinusIJ, i] == 0)[0]
                if indNotConnected.size == 0:
                    failAccum += 1
                    #print('Failed to find k, failAccum is %d'%failAccum)
                else:
                    k = np.random.choice(indMinusIJ[indNotConnected])

                    # Finding d -if it exists, it should be connected to k but not to j
                    # the d candidate should not be the same as i or j (or k too but we made sure it is not there)
                    indCon = np.where(A[indMinusIJ, k] > 0)[0]
                    indUncon = np.where(A[indMinusIJ, j] == 0)[0]
                    # pick all the candidates d by taking the intersection
                    dCandInd = np.intersect1d(indCon, indUncon)
                    dCand = indMinusIJ[dCandInd]

                    if dCand.size == 0:
                        failAccum += 1
                        #print('Failed to find d, failAccum is %d'%failAccum)

                    else:
                        d = np.random.choice(dCand)
                        if A[i, k] > 0 or A[j, d] > 0:
                            print('there is something wrong, they should be zero')
                        if A[i, j] == 0 or A[k, d] == 0:
                            print('there is somehting wrong. they should not be zero')
                        A[i, k] = A[i, j]  # I could have put 1s, but maybe we generalize to weighted graphs too
                        A[k, i] = A[i, j]
                        A[j, d] = A[k, d]
                        A[d, j] = A[k, d]
                        A[i, j] = 0
                        A[j, i] = 0
                        A[k, d] = 0
                        A[d, k] = 0
                        failAccum = 0
                        rewire += 1
                        #print('number of rewirings is %d'%rewire)

                        if rewire >= swaps:
                            flagStay = False

        if failAccum >= threshConsecFail:
            flagStay = False

    return A, rewire

# RANDOMIZEWEIGHTSKEEPTOPOLOGY gets the random matrix for the calculation of the normalized WEIGHTED rich club coeff.
# The random matrix has the same topology as the original network but reshuffled edge weights
# Alstott, J., Panzarasa, P., Rubinov, M., Bullmore, E. T., & Vértes, P. E. (2014). A unifying framework for measuring weighted rich clubs. Scientific reports, 4, 7258.
# INPUT:
# Adj: the weighted adjacency matrix
# OUTPUT:
# A: the reshuffled random adjacency matrix


def randomizeWeightsKeepTopology(Adj):

    A = Adj.copy()

    indNonZero = np.where(A > 0)  # finds the indices of the nonzero elements -weights

    # stores a reshuffled version of the weights in the vec weights
    weights = np.zeros(len(indNonZero[0]))
    weights = A[indNonZero]

    newWeights = np.random.permutation(weights)
    A[indNonZero] = newWeights

    return A


# GETNORMALIZEDRICHCLUBCOEF
# INPUT:
# A: the adjacency matrix
# k: a scalar positive integer value indicating that the nodes consisting the rich club should have degree greater than k
# binaryFlag: if True we calculate the binary or topological rich club coeffs, if False we calculate the weighted one
# verbose: defaults to True
# for the binary or topological normalized rich club check the following paper:
# Towlson, Emma K., et al. "The rich club of the C. elegans neuronal connectome." Journal of Neuroscience 33.15 (2013): 6380-6387.
# for the weighted normalized rich club check this one:
# Alstott, J., Panzarasa, P., Rubinov, M., Bullmore, E. T., & Vértes, P. E. (2014). A unifying framework for measuring weighted rich clubs. Scientific reports, 4, 7258.
# OUTPUT:
# phiK: unormalized rich club coeff
# phiKRand: rich club coef of the control or random network
# phiKNorm: normalized rich club coeff --> phiK/phiKRand
# mK: number of connections between rich club nodes
# nK: number of nodes that are rich club
def getNormalizedRichClubCoef(A, k, binaryFlag, verbose=True):

    if binaryFlag == True:
        (phiK, mK, nK) = getRichClubCoef(A, k, binaryFlag)
        swaps = 100 * mK
        Arew, rewire = randomizeKeepDegDistr(A, swaps)
        (phiKRand, mKRand, nKRand) = getRichClubCoef(Arew, k, binaryFlag)
    else:
        (phiK, mK, nK, totalStr, maxStr) = getRichClubCoef(A, k, binaryFlag)
        Arew = randomizeWeightsKeepTopology(A)
        (phiKRand, mKRand, nKRand, totalStrRand, maxStrRand) = getRichClubCoef(Arew, k, binaryFlag)

    phiKNorm = phiK / phiKRand

    if verbose == True:
        print('For the rich club, the number of connections are %d and the number of nodes are %d' % (mK, nK))
        print('For the rewired rich club, the number of connections are %d and the number of nodes are %d' % (mKRand, nKRand))

        if binaryFlag is not True:
            print('The pairs should be the same above')
            print('for the rich club the total weight of connections is %f and the max weight the could have is %f' % (totalStr, maxStr))
            print('for the rewired rich club the total weight of connections is %f and the max weight the could have is %f' % (totalStrRand, maxStrRand))
        else:
            print('asked for %d swaps, got %d swaps' % (swaps, rewire))

        print('The unormalized rich club coef is %f' % phiK)
        print('The rich club coef for the rewired control network is %f' % phiKRand)
        print('The normalized rich club coef is %f' % phiKNorm)

    return phiK, phiKRand, phiKNorm, mK, nK
