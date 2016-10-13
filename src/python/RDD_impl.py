from __future__ import print_function

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

# def deltaF(x, AtA, ytA, lamda):
#     """
#     1st Derivative of LASSO
#     """
#     retVal = (np.dot(x.transpose(),AtA) - ytA) + lamda;
#     return retVal

def F(x,A,y,lamda):
    """
    LASSO function
    """
    return 0.5 * np.square(np.linalg.norm(np.dot(A,x) - y)) + lamda*np.linalg.norm(x,1)

if __name__ == "__main__":
    data = sio.loadmat("../../data/Mug32_singlepixcam.mat",mat_dtype=True)
    lamda = 0.05 # lamda ("lambda") for Mug32_singlepixcam = 0.05
    y = data['y']
    A = data['A']

    N,d = A.shape
    print(N)

    sc = SparkContext(appName='pySparkShotgun', master='local')

    ## Parallelizing rowise
    # A = sc.parallelize(xrange(N)).map(lambda i: (A[i,:],y[i])).persist()

    # # Normalizing A
    # norm_const = np.sqrt(A.map(lambda (a,y): np.square(a)).reduce(lambda a,b: a+b))
    # norm_const = sc.broadcast(norm_const)
    # A = A.map(lambda (a,y): (np.divide(a,norm_const.value),y))

    # # Double the dimensions of A
    # A = A.map(lambda (a,y): (np.append(a,-a),y))
    # 
    # print(len(A.collect()[0][0]))

# In this implementation A fits into memory
    A = normc(A) 
    # If A doesn't fit then:
    # 1. Load A as a columnwise (Normalize as you load) RDD
    # 2. Maintain column ids for the next steps.

    # Initialize an empty x
    x = np.zeros(2*d)
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
    data = sc.parallelize(xrange(2*d)).\
            map(lambda cid: (cid,
                            np.dot(np.transpose(A[:,cid % d]),A),
                            np.dot(np.transpose(y),A[:,cid % d]))).\
            persist() #here cid is the column index.
    # NOTE: When computing AtA if A doesn't fit into memory we:
    # 1. Compute individual elements of the AtA matrix.
    # 2. Reconstruct/Reduce the elements of the matrix into RDDs of columns.
    # 3. Proceed with the next steps. Remember to maintain the column ids for the next step.

    # Constants
    beta = sc.broadcast(1)
    lamda = sc.broadcast(0.05)

    maxiters = 1000

    def deltaF((cid, AtAi, ytAi)):
        if cid < d:
            i = 1
        else:
            i = -1
        AtAi_2d =  np.concatenate((i*AtAi, -1*i*AtAi))
        return (cid, np.dot(x.transpose(), AtAi_2d) - ytAi + lamda.value)
        

    for i in range(10):
        P_vals=[1,2,4,6,8,10,20,30,40,50,60,70,80,90,100,110]
        for P in P_vals:
            for iter in xrange(maxiters):
                randIdxs = np.random.permutation(2*d)
                randPidxs = randIdxs[0:P]

                # Get the data corresponding to the P indexes.
                # NOTE: Problem !! cid(s) in data_sub are always
                #       less than d. So the map with deltaF will
                #       not work. !
                data_sub = data.filter(lambda (cid, AtAi, ytAi): cid in randPidxs or cid+d in randPidxs)
                # updates = data_sub.map(lambda (cid, AtAi, ytAi): (cid, np.dot(x.transpose(), AtAi) - ytAi + lamda.value)).collect()
                updates = np.array(data_sub.map(deltaF).collect())
                idxs = np.array(updates[:,0],int)
                # x[idxs] += np.maximum(-1*x[idxs], -1*updates[:,1]/beta.value)
                x[idxs] += np.max(np.array([-1*x[idxs], -1*updates[:,1]/beta.value]),axis=0)

                if iter % 100 == 0:
                    # print(idxs)
                    # print(x[x!=0])
                    x_1d = x[d:] - x[:d]
                    # print(x_1d[x_1d!=0])
                    # # Following line would change if A doesn't fit in memory
                    print("Iter {} NormVal {}".format(iter, F(np.reshape(x_1d,(d,1)), A, y, lamda.value))) 
    # # Normalizing A
    # data = data.map(lambda (cid,a,y):\
    #         (cid,a/np.sqrt(np.sum(np.square(a))),y))
    
    # For sanity check otherwise operations don't execute.
    data.collect()
