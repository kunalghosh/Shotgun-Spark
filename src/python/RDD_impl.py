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
    # return 0.5 * np.square(np.linalg.norm(np.dot(np.concatenate((A,-A),axis=1),x) - y)) + lamda*np.sum(x)

if __name__ == "__main__":
    data = sio.loadmat("../../data/Mug32_singlepixcam.mat",mat_dtype=True)
    lamda = 0.05 # lamda ("lambda") for Mug32_singlepixcam = 0.05
    y = data['y']
    A = data['A']

    N,d = A.shape
    print(N)

    # sc = SparkContext(appName='pySparkShotgun', master='local')
    sc = SparkContext(appName='pySparkShotgun')

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
    data = sc.parallelize(xrange(d)).\
            map(lambda cid: (cid,
                            np.dot(np.transpose(A[:,cid]),A),
                            np.dot(np.transpose(y),A[:,cid]))).\
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
        P_vals=[3,1,2,4,6,8,10,20,30,40,50,60,70,80,90,100,110]
        for P in P_vals:
            for iter in xrange(maxiters):
                randIdxs = np.random.permutation(2*d)
                randPidxs = randIdxs[0:P]
                # print(randPidxs[randPidxs <= d])
                # print(randPidxs[randPidxs > d])
                # print(randPidxs) 
                # Get the data corresponding to the P indexes.
                # NOTE: Problem !! cid(s) in data_sub are always
                #       less than d. So the map with deltaF will
                #       not work. !
                # data_sub = data.filter(lambda (cid, AtAi, ytAi): cid in randPidxs or cid+d in randPidxs)
                # updates = data_sub.map(lambda (cid, AtAi, ytAi): (cid, np.dot(x.transpose(), AtAi) - ytAi + lamda.value)).collect()
                
                # We do the updates in two parts
                # for column ids in randPidxs less than d
                data_sub_lt_d = data.filter(lambda (cid, AtAi, ytAi): cid in randPidxs)
                updates_lt_d = data_sub_lt_d.map(deltaF).collect()

                # for column ids in randPidxs greater than d
                data_sub_gt_d = data.filter(lambda (cid, AtAi, ytAi): cid+d in randPidxs)\
                                    .map(lambda (cid, AtAi, ytAi): (cid+d, AtAi, ytAi)) # update the column ids
                updates_gt_d = data_sub_gt_d.map(deltaF).collect()

                # Sanity check for P = 1 one of the data_sub_--_d should be empty
                # Count of Columns (with cid < d and cid > d) must add up to P. Since cid(s) come from randPidxs
                assert P == data_sub_lt_d.count() + data_sub_gt_d.count() , "(# of cid(s) < d) {} + {} (# of cid(s) > d) != {} (= P)".format(data_sub_lt_d.count(),data_sub_gt_d.count(),P)
                assert P == len(updates_lt_d) + len(updates_gt_d) , "(# of updates from cid(s) < d) {} + {} (# of updates from cid(s) > d)!= {} (= P)".format(len(updates_lt_d),len(updates_gt_d),P)

                # print(updates_lt_d)
                # print(updates_gt_d)
                
                updates = np.array(updates_lt_d+updates_gt_d)
                idxs = np.array(updates[:,0],int)
                # print(idxs)
                # print(np.max(np.array([-1*x[idxs], -1*updates[:,1]/beta.value]),axis=0))
                x[idxs] += np.max(np.array([-1*x[idxs], -1*updates[:,1]/beta.value]),axis=0)

                if iter % 100 == 0:
                    # print(idxs)
                    # print(x[x!=0])
                    x_1d = x[d:] - x[:d]
                    print(x_1d[x_1d!=0])
                    # # Following line would change if A doesn't fit in memory
                    # print("Nonzero x {}".format(np.nonzero(x_1d))) 
                    print("Iter {} NormVal {}".format(iter, F(np.reshape(x_1d,(d,1)), A, y, lamda.value))) 
                    # print("Iter {} NormVal {}".format(iter, F(np.reshape(x,(2*d,1)), A, y, lamda.value))) 
    # # Normalizing A
    # data = data.map(lambda (cid,a,y):\
    #         (cid,a/np.sqrt(np.sum(np.square(a))),y))
    
    # For sanity check otherwise operations don't execute.
    data.collect()
