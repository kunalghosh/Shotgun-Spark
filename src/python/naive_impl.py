from __future__ import print_function

import numpy as np
import scipy.io as sio
import pprint

# from pyspark import SparkContext
# from pyspark.ml.linalg import DenseVector



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

if __name__ == "__main__":
    data = sio.loadmat("../../data/Mug32_singlepixcam.mat")
    lamda = 0.05 # lamda ("lambda") for Mug32_singlepixcam = 0.05
    y = data['y']
    A = data['A']

    # sc = SparkContext(appName="Shotgun")
    # y = sc.parallelize(y)
    # A = sc.parallelize(A)

    N,d = A.shape
    x_org = np.zeros((d,1))
    condition = True

    x = x_org
    A = normc(A)
    AtA = np.dot(A.transpose(),A)
    ytA = np.dot(y.transpose(),A)

    # Initialization
    iter = 0
    P = 20
    # rho = max(eigs(AtA)) # TODO: Implement eigs
    # P_opt = d/rho; # TODO: implement this once we have the rho

    beta = 1 # for LASSO
    x_opt_collection = []
    for i in range(10):
    # for i in range(2):
        # Run shotgun 10 times
        iterations=[]
        P_Val=[1,2,4,6,8,10,20,30,40,50,60,70,80,90,100,110];
        print("Nonzero x {}".format(np.nonzero(x)))
        for p in P_Val:
            normVal = []
            iter = 0
            condition = True
            x=np.zeros((d,1))
            print("Nonzero x {}".format(np.nonzero(x_org)))
            while condition:
                iter += 1
                randIdxs = np.random.permutation(d)
                randPidxs = randIdxs[0:P]
                x_subset = x[randPidxs]

                deltaF_vals = deltaF(x,AtA,ytA,lamda)
                # print(deltaF_vals)
                deltaF_subset = deltaF_vals[(0,randPidxs)]
                # print(deltaF_subset)
                # P parallel updates
                # NOTE: Not parallel in this version
                for j in range(P):
                    delta_xj = np.max([-1 * x_subset[j], -1*deltaF_subset[j]/beta])
                    # print("x_subset {} deltaF_subset {} delta_xj {}".format(-1 * x_subset[j],-1*deltaF_subset[j]/beta, delta_xj))
                    x_subset[j] = x_subset[j] + delta_xj

                x[randPidxs] = x_subset

                # normVal.append(F(x,A,y,lamda) - F(x_opt,A,y,lamda))
                condition = iter < 1000
                # normVal.append( (F(x) - F(x_opt2.beta)) /F(x_opt2.beta))
                # TODO: How to calculate x_opt2.beta in this case ?

                if iter % 100 == 0:
                    # print("Iterations {} NormVal {} x-non_zero {}".format(iter, F(x,A,y,lamda), np.nonzero(x.transpose())))
                    print("Iterations {} NormVal {}".format(iter, F(x,A,y,lamda)))

            iterations.append({p:(iter,x)})
            print("i {} p {} iter {}".format(i,p,iter))
        x_opt_collection.append({i : iterations})
    pprint.pprint(x_opt_collection)
