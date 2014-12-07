# Discrete Martrix Factorizatin With Cramer Risk
Jianbo Ye (c) 2014-2015

This is the experimental code for paper:

Jianbo Ye, Top-k Probability Estimation Using Discrete Martrix Factorizatin: A Cram\"er Risk Minimization Approach (to appear, 2015)

Input 
 - A sparse matrix with values taking discrete values from 1 .. M, each row is a user, and each column is an item. 
 - A representation dimension for user and item factors
  
Output: 
 - A probability matrix for each cell (i,j) with a nonzero probability for discrete value from 0 ... (M+1), where 0 and (M+1) are considering as the "extrame values". One can either use the expected value as predictions or rank each row using extrame values.
 - For each item j, the estimation of its Top-k probability in the preference list of user i can vbe computed. 


