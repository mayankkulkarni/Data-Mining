# Data-Mining
Recommendation Systems, frequent itemsets and social network analysis built using Python and Apache Spark

RECOMMENDATION SYSTEMS
Recommendation Systems built using "ml-latest-small" Movielens dataset for user movie pairs.

USER BASED CF
Number of predictions : 20,256
Running time : 40s
RMSE : 0.945736552229

ITEM BASED CF
Number of predictions : 20,256
Running time : 20s
RMSE : 0.974781926478
Offline pre-processing of similarity between items using Jaccard Similarity takes time 120s

MODEL BASED CF
Algorithm used - Alternating Least Squares algorithm
Number of Predictions : 4,046,331
Running time : 1639s
RMSE : 0.821399395235
ALS is built in algorithm in Apache Spark's ML library - MLLib. 

I used mean imputation to handle data sparsity. 
Possible improvements would be to add case amplification in order to reduce noise and hence improve RMSE.
