from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.io as sio
import pprint

from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, explode, struct, lit

def normc(x):
    """ 
    Normalize such that sum of squares of columns = 1
    """
    return x / np.sqrt(np.square(x).sum(axis=0))

def deltaF(x, AtA, ytA, lamda):
    """
    1st Derivative of LASSO
    """
    retVal = (np.dot(x.transpose(),AtA) - ytA) + lamda;
    return retVal

def F(x,A,y,lamda):
    """
    LASSO function
    """
    return 0.5 * np.square(np.linalg.norm(np.dot(A,x) - y)) + lamda*np.linalg.norm(x,1)

def maxEigenValue(AtA, max_iters=100, eps=10**-2 ):
    """
    Given a symmetric (column major) matrix, returns its
    maximum eigen value using power iteration method.

    params: 
    """
    d = AtA.count() # the number of dimensions
    max_eig = None
    v = np.random.random(d)
    for _ in xrange(max_iters):
        # print(v)
        v_new = AtA.map(lambda (AtAi): np.dot(AtAi,v)).collect()
        norm = np.linalg.norm(v_new,2)
        v_new /= norm
        # print(v_new)
        max_eig = norm
        l2_diff = np.linalg.norm(v_new-v,2)
        # print("Debug maxEig : {}, l2_diff : {}".format(max_eig,l2_diff))
        if l2_diff < eps:
            break
        v = v_new

    return max_eig

if __name__ == "__main__":
    data = sio.loadmat("../../data/Mug32_singlepixcam.mat",mat_dtype=True)
    y = data['y']
    A = data['A']

    N,d = A.shape

    # sc = SparkContext(appName='pySparkShotgun', master='local')
    sc = SparkContext(appName='pySparkShotgun')

# In this implementation A fits into memory
    A = normc(A) 
    # If A doesn't fit then:
    # 1. Load A as a columnwise (Normalize as you load) RDD
    # 2. Maintain column ids for the next steps.

    # Initialize an empty x
    x = np.zeros(d)
    # Calculating A_newTA_new where A_new = [A, -A] 
    # So A_newTA_new is 
    # |<-d->
    # | AtA|-AtA |
    # |----+-----|
    # |-AtA| AtA |
    # So we just compute AtA once
    # if column id > d (dimension of AtA)
    # concatenate(-AtA[:,id],AtA[:,id]) columnwise
    # else
    # concatenate(AtA[:,id],-AtA[:,id]) columnwise
    ## aTa = sc.parallelize(xrange(d)).map(lambda cid: np.dot(np.transpose(A[:,cid]),A))

    # Parallelizing column wise
    data = sc.parallelize(xrange(d)).\
            map(lambda cid: (cid,
                            np.dot(np.transpose(A[:,cid]),A),
                            np.dot(np.transpose(y),A[:,cid]))).\
            persist() #here cid is the column index.
    
    AtA = data.map(lambda (cid, AtAi, ytAi): AtAi)
    max_eig = maxEigenValue(AtA)
    print("Maximum Eigen value : ",max_eig)

    P_opt = int(d/max_eig) # maxeig is the spectral radius (rho) in the paper. 
    print("Optimal number of parallel updates : ",P_opt)

    # NOTE: When computing AtA if A doesn't fit into memory we:
    # 1. Compute individual elements of the AtA matrix.
    # 2. Reconstruct/Reduce the elements of the matrix into RDDs of columns.
    # 3. Proceed with the next steps. Remember to maintain the column ids for the next step.

    # Constants
    beta = sc.broadcast(1)
    lamda = sc.broadcast(0.05) # lamda ("lambda") for Mug32_singlepixcam = 0.05

    maxiters = 1000
    print(data.count())

    for i in range(10):
        P_vals=[P_opt]
        # NOTE : YAY !! if P_vals = P_opt/2 (400 iters) then we should take twice the number
        # of iterations as P = P_opt (200 iters). Exactly that happens :D and here P_opt = 158 (wOOt !)
        # and runs fine on my laptop :P
        for P in P_vals:
            x = np.zeros(d)
            for iter in xrange(maxiters):
                randIdxs = np.random.permutation(d)
                randPidxs = randIdxs[0:P]

                # Get the data corresponding to the P indexes.
                data_sub = data.filter(lambda (cid, AtAi, ytAi): cid in randPidxs)
                updates = np.array(data_sub.map(lambda (cid, AtAi, ytAi): (cid, np.dot(x.transpose(), AtAi) - ytAi + lamda.value)).collect())
                
                idxs = np.array(updates[:,0],int)
                x[idxs] += np.max(np.array([-1*x[idxs], -1*updates[:,1]/beta.value]),axis=0)

                if iter % 100 == 0:
                    print(x[x!=0])
                    # # Following line would change if A doesn't fit in memory
                    print("Iter {} NormVal {}".format(iter, F(np.reshape(x,(d,1)), A, y, lamda.value))) 
    
    # For sanity check otherwise operations don't execute.
    data.collect()
