# Discrete Martrix Factorization with Cramer Risk Minimization [![DOI](https://zenodo.org/badge/19103/bobye/dmfCramer.svg)](https://zenodo.org/badge/latestdoi/19103/bobye/dmfCramer)
Jianbo Ye (c) 2014-2015


This is an experimental code.
Input 
 - A sparse matrix with values taking discrete values from 1 .. M, each row is a user, and each column is an item. 
 - A representation dimension for user and item factors (default: 10)
  
Output: 
 - A probability matrix for each cell (i,j) with a nonzero probability for discrete value from 0 ... (M+1), where 0 and (M+1) are considering as the "extrame values". One can either use the expected value as predictions or rank each row using extrame values.


## Reference
 - Dataset: http://grouplens.org/datasets/movielens/
