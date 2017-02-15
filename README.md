# "Shotgun" Spark
Implementation of the Parallel Coordinate Descent for L1-Regularized Loss Minimization ( [arxiv](https://arxiv.org/abs/1105.5379),   [code](https://github.com/akyrola/shotgun) from authors of the paper ) in Spark.

The [report](https://github.com/kunalghosh/Shotgun-Spark/blob/master/report/Report_final.pdf) goes into the details of how to run the Spark implementation on a SLURM (slurm.shedmd.org) based linux cluster.

# Dependencies:
## For the Python Implementation
  1. Apache Spark : 2.0.1
  2. PySpark
  3. Python2.7
  4. NumPy
  5. Scipy (For loading the binary Matlab datafile used by the Paper)
## For the Matlab Implementation
Apart from a working installation of Matlab, there is no other dependency.
  
