# Data-Mining
Recommendation Systems built using Python and Apache Spark<br>
<br>
RECOMMENDATION SYSTEMS<br>
Recommendation Systems built using "ml-latest-small" Movielens dataset for user movie pairs.
<br>
<br>
USER BASED CF<br>
Number of predictions : 20,256<br>
Running time : 40s<br>
RMSE : 0.945736552229<br>
<br>
ITEM BASED CF<br>
Number of predictions : 20,256<br>
Running time : 20s<br>
RMSE : 0.974781926478<br>
Offline pre-processing of similarity between items using Jaccard Similarity takes time 120s<br>
<br>
MODEL BASED CF<br>
Algorithm used - Alternating Least Squares algorithm<br>
Number of Predictions : 4,046,331<br>
Running time : 1639s<br>
RMSE : 0.821399395235<br>
ALS is built in algorithm in Apache Spark's ML library - MLLib.<br>
<br>
I used mean imputation to handle data sparsity.<br> 
Possible improvements would be to add case amplification in order to reduce noise and hence improve RMSE.<br>
