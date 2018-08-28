from pyspark import SparkConf, SparkContext
import sys
import time

conf = SparkConf()
s = SparkContext(conf=conf)


def create_utility_matrix(utilityrdd):
    baskets = utilityrdd.map(lambda x: (x[0][0],{(x[0][1],x[1])})).reduceByKey(lambda x,y: x.union(y))
    return baskets


def pearson_coeff(x,a,u):
    w=0
    a = dict(a)
    u = dict(u)
    rated_a = set(a.keys())
    rated_u = set(u.keys())
    corated_items = rated_u.intersection(rated_a)
    count = len(corated_items)
    if not count:
        return (x,w)
    sum_a=0
    sum_u=0
    for i in corated_items:
        sum_a+=a[i]
        sum_u+=u[i]
    avg_a = sum_a/count
    avg_u = sum_u/count
    num = 0
    den_a=0
    den_b=0
    for i in corated_items:
        t1 = a[i]-avg_a
        t2 = u[i]-avg_u
        num+=t1*t2
        den_a += t1**2
        den_b += t2**2
    if den_a and den_b:
        w = num/((den_a**0.5)*(den_b**0.5))
    return (x,w)


def avg_rating_active_user(x):
    sum=0
    for i in x[1]:
        sum+=i[1]
    if len(x[1])!=0:
        return (x[0],sum/len(x[1]))
    else:
        return (x[0],0)


def create_user_movie(x):
    movies = set([])
    for i in x[1]:
        movies.add(i[0])
    return (x[0],movies)


def create_user_to_relate(x,user_movie):
    related_users = set([])
    for i in user_movie:
        if i[1].intersection(x[1]):
            related_users.add((x[0],i[0]))
    return related_users


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


def predict_rating(i,a_avg_rating,movie_users, utility_matrix, pearson_coeffs):
    num = 0
    den = 0
    predicted_rating = 0
    if i[0] in a_avg_rating:
        predicted_rating = a_avg_rating[i[0]]
    if i[1] in movie_users:
        item_raters = movie_users[i[1]].keys()
        for j in item_raters:
            item_ratings = dict(utility_matrix[j])
            del item_ratings[i[1]]
            sum = 0
            for k in item_ratings:
                sum += item_ratings[k]
            if len(item_ratings) != 0:
                raters_avg = sum / len(item_ratings)
            else:
                raters_avg = 0
            num += (movie_users[i[1]][j] - raters_avg) * pearson_coeffs[(i[0], j)]
            den += abs(pearson_coeffs[(i[0], j)])
        if den != 0:
            predicted_rating += (num / den)
        if predicted_rating > 5:
            predicted_rating -= (num / den)
    if i[0] not in a_avg_rating and i[1] not in movie_users:
        predicted_rating = 2.5
    return (i,predicted_rating)


def main(ratingfilename, testingfilename):
    ratingrdd = s.textFile(ratingfilename).repartition(8)
    prediction_items = s.textFile(testingfilename).repartition(8)
    ratingrdd = ratingrdd.map(lambda x: x.split(',')).filter(lambda x: x[0] != u'userId').map(lambda x: ((int(x[0]),int(x[1])),float(x[2])))
    prediction_items = prediction_items.map(lambda x: x.split(',')).filter(lambda x: x[0] != u'userId').map(lambda x: ((int(x[0]),int(x[1])),1))
    verifyrdd = prediction_items.join(ratingrdd).map(lambda x: (x[0], x[1][1]))

    utilityrdd = ratingrdd.subtractByKey(prediction_items)
    utilityrdd = utilityrdd.repartition(8)

    items_to_predict = prediction_items.keys().map(lambda x: (x[0],{x[1]})).reduceByKey(lambda x,y: x.union(y))
    utility_matrix = create_utility_matrix(utilityrdd)
    users_to_predict = items_to_predict.keys().collect()
    a_avg_rating_rdd = utility_matrix.filter(lambda x: x[0] in users_to_predict)
    a_avg_rating = a_avg_rating_rdd.map(lambda x: avg_rating_active_user(x)).collectAsMap()
    user_movie = utility_matrix.map(lambda x: create_user_movie(x)).collect()
    users_to_relate = items_to_predict.flatMap(lambda x: create_user_to_relate(x,user_movie))
    utility_matrix = utility_matrix.collectAsMap()
    pearson_coeffs = users_to_relate.map(lambda x: pearson_coeff(x, utility_matrix[x[0]], utility_matrix[x[1]])).collectAsMap()
    movie_users = utilityrdd.map(lambda x: (x[0][1],{(x[0][0],x[1])})).reduceByKey(lambda x,y: x.union(y))
    movie_users = movie_users.map(lambda x: (x[0],dict(x[1]))).collectAsMap()
    prediction_items = prediction_items.keys()
    predictions = prediction_items.map(lambda x: predict_rating(x,a_avg_rating,movie_users,utility_matrix,pearson_coeffs)).collect()
    predictions_rdd = s.parallelize(predictions)
    ratesAndPreds = verifyrdd.join(predictions_rdd)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    prediction_baseline = create_baseline(ratesAndPreds).collect()
    with open("Mayank_Kulkarni_UserBasedCF.txt",'w') as file:
        for i in predictions:
            file.write(','.join(str(x) for x in i[0]))
            file.write(","+str(i[1]))
            file.write("\n")
    for i in prediction_baseline:
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
