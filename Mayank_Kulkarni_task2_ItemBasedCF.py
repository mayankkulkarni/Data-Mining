from pyspark import SparkConf, SparkContext
from itertools import combinations
import sys
import time

conf = SparkConf()
s = SparkContext(conf=conf)


def count_movies_users(textRDD):
    movies = textRDD.map(lambda x: (int(x[1]),1)).reduceByKey(lambda x,y: x+y).keys()
    numMovies = movies.count()
    users = textRDD.map(lambda x: (int(x[0]),1)).reduceByKey(lambda x,y: x+y).keys()
    numUsers = users.count()
    return numMovies, numUsers


def create_characteristic_matrix(textRDD,numMovies,numUsers):
    charMatrix = [[0]*numUsers for i in range(numMovies)]
    movies = sorted(textRDD.map(lambda x: (int(x[1]),1)).reduceByKey(lambda x,y: x+y).keys().collect())
    baskets = textRDD.map(lambda x: (int(x[0]),{int(x[1])})).reduceByKey(lambda x,y: x.union(y)).collect()
    for i in baskets:
        for j in i[1]:
            charMatrix[movies.index(j)][i[0]-1] = 1
    return movies, baskets, charMatrix


def create_signatures(textRDD, movies, numMovies, numUsers):
    movie_users = textRDD.map(lambda x: (int(x[1]),{int(x[0])})).reduceByKey(lambda x,y: x.union(y)).persist()
    a = 17
    b = 19
    m = numUsers
    n = 12
    movies = s.broadcast(movies)
    sig_matrix = [[9999]*n for i in range(numMovies)]
    for i in range(1,n+1):
        hf = lambda x: (i*(a*x + b))%m
        sig_movies = movie_users.map(lambda x: (movies.value.index(x[0]),min(map(lambda x: hf(x),x[1])))).collect()
        for j in sig_movies:
            if sig_matrix[j[0]][i-1] > j[1]:
                sig_matrix[j[0]][i-1] = j[1]
    return n,sig_matrix


def generate_pairs(x):
    return frozenset(combinations(x[1],2))


def candidate_pairs(n,sig_matrix):
    b = 6
    r = n/b
    candidates = frozenset([])
    for i in range(b):
        band = [tuple(sig_matrix[k][2*i:2*i+r]) for k in range(len(sig_matrix))]
        band = s.parallelize(band).zipWithIndex()
        candidate_sigs = band.map(lambda x: (x[0],[x[1]])).reduceByKey(lambda x,y: x+y).filter(lambda x: len(x[1])>1)
        candidates_band = candidate_sigs.map(lambda x: (1,generate_pairs(x))).reduceByKey(lambda x,y: x.union(y)).values().collect()
        candidates = candidates.union(candidates_band[0])
    return candidates


def compare_candidates(x,threshold,charMatrix,movies, numUsers):
    intersect = 0
    a=0
    b=0
    for i in range(numUsers):
        if charMatrix[x[0]][i] == 1 and charMatrix[x[1]][i] == 1:
            intersect+=1
        if charMatrix[x[0]][i] == 1:
            a+=1
        if charMatrix[x[1]][i] == 1:
            b+=1
    union = a+b - intersect
    similarity = float(intersect)/float(union)
    if similarity>=threshold:
        return ((movies[x[0]],movies[x[1]]),similarity)


def similar_items(candidates, threshold, charMatrix, movies, numUsers):
    candidates = s.parallelize(candidates)
    items = candidates.map(lambda x: compare_candidates(x,threshold,charMatrix, movies, numUsers)).filter(lambda x: x!=None)
    return items


def generate_similar_items_lsh(utilityrdd):
    similarity_threshold = 0.5
    numMovies, numUsers = count_movies_users(utilityrdd)
    movies, baskets, charMatrix = create_characteristic_matrix(utilityrdd, numMovies, numUsers)
    n, sig_matrix = create_signatures(utilityrdd, movies, numMovies, numUsers)
    candidates = candidate_pairs(n,sig_matrix)
    sim_items = similar_items(candidates, similarity_threshold, charMatrix, movies, numUsers).collect()
    return sim_items


def item_similar_items(x,prediction_items):
    items = []
    if x[0][0] in prediction_items:
        items+=[(x[0][0],[(x[0][1],x[1])])]
    if x[0][1] in prediction_items:
        items+=[(x[0][1],[(x[0][0],x[1])])]
    return items


def create_class(x):
    if x>=0 and x<=1:
        return (1,1)
    elif x>=1 and x<=2:
        return (2,1)
    elif x>=2 and x<=3:
        return (3,1)
    elif x>=3 and x<=4:
        return (4,1)
    elif x>=4:
        return (5,1)


def create_baseline(ratesAndPreds):
    return ratesAndPreds.map(lambda x: create_class(abs(x[1][0]-x[1][1]))).reduceByKey(lambda x,y: x+y)


def create_lsh_prediction(i,user_movie_ratings,lsh_similar_items):
    user_items = user_movie_ratings[i[0]]
    user_items = dict(user_items)
    num = 0
    den = 0
    if i[1] in lsh_similar_items:
        sim_items = lsh_similar_items[i[1]]
        sim_items = dict(sim_items)
        for j in user_items:
            w = 0
            if j in sim_items:
                w = sim_items[j]
            num += (w * user_items[j])
            if w != 0:
                den += w
        if den != 0:
            return (i, num / den)
    if i[1] in lsh_similar_items or den == 0:
        sum = 0
        for j in user_items:
            sum += user_items[j]
        return (i, (sum / len(user_items)))


def main(ratingfilename, testingfilename):
    ratingrdd = s.textFile(ratingfilename).repartition(8)
    prediction_pairs = s.textFile(testingfilename).repartition(8)
    ratingrdd = ratingrdd.map(lambda x: x.split(',')).filter(lambda x: x[0] != u'userId').map(lambda x: ((int(x[0]),int(x[1])),float(x[2])))
    prediction_pairs = prediction_pairs.map(lambda x: x.split(',')).filter(lambda x: x[0] != u'userId').map(lambda x: ((int(x[0]),int(x[1])),1))
    verifyrdd = prediction_pairs.join(ratingrdd).map(lambda x: (x[0], x[1][1]))

    utilityrdd = ratingrdd.subtractByKey(prediction_pairs)
    utilityrdd = utilityrdd.repartition(8)
    utilityrdd_keys = utilityrdd.keys()
    lsh_similar_items = generate_similar_items_lsh(utilityrdd_keys)
    lsh_similar_items_rdd = s.parallelize(lsh_similar_items)
    prediction_items = prediction_pairs.map(lambda x: (x[0][1],x[0][0])).keys().collect()
    lsh_similar_items_rdd = lsh_similar_items_rdd.filter(lambda x: x[0][0] in prediction_items or x[0][1] in prediction_items).flatMap(lambda x: item_similar_items(x,prediction_items)).reduceByKey(lambda x,y: x+y)
    lsh_similar_items = lsh_similar_items_rdd.collectAsMap()
    prediction_pairs = prediction_pairs.keys()
    user_movie_ratings = utilityrdd.map(lambda x: (x[0][0],{(x[0][1],x[1])})).reduceByKey(lambda x,y: x.union(y)).collectAsMap()
    predictions_lsh_rdd = prediction_pairs.map(lambda x: create_lsh_prediction(x,user_movie_ratings,lsh_similar_items))
    ratesAndPreds = verifyrdd.join(predictions_lsh_rdd)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    prediction_baseline_lsh = create_baseline(ratesAndPreds).collect()
    with open("Mayank_Kulkarni_ItemBasedCF.txt",'w') as file:
        predictions_lsh = sorted(predictions_lsh_rdd.collect())
        for i in predictions_lsh:
            file.write(','.join(str(x) for x in i[0]))
            file.write(","+str(i[1]))
            file.write("\n")
    for i in prediction_baseline_lsh:
        if i[0] != 5:
            print ">=" + str(i[0] - 1) + " and <" + str(i[0]) + ": " + str(i[1])
        else:
            print ">=" + str(i[0] - 1) + ": " + str(i[1])
    print "RMSE: " + str(MSE ** 0.5)


if __name__ == "__main__":
    t = time.time()
    ratingFileName = sys.argv[1]
    testingFileName = sys.argv[2]
    main(ratingFileName, testingFileName)
    print "Time: ", time.time()-t
