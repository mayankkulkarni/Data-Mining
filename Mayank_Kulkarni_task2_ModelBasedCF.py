from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
#import pyspark.mllib.recommendation.MatrixFactorizationModel as MatrixFactorizationModel
#import pyspark.mllib.recommendation.Rating s
import sys
import time

conf = SparkConf()
s = SparkContext(conf=conf)


def train_model(trainingrdd):
    rank = 8
    numIterations = 8
    model = ALS.train(trainingrdd, rank, numIterations, 0.1, blocks=8)
    return model


def test_model(testingrdd, model):
    testingrdd = testingrdd.keys()
    predictions = model.predictAll(testingrdd).map(lambda r: ((r[0], r[1]), r[2]))
    return predictions

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


def main(ratingfilename, testingfilename):
    ratingrdd = s.textFile(ratingfilename)
    ratingrdd = ratingrdd.repartition(8)
    testingrdd = s.textFile(testingfilename)
    testingrdd = testingrdd.repartition(8)
    ratingrdd = ratingrdd.map(lambda x: x.split(',')).filter(lambda x: x[0] != u'userId').map(lambda x: ((int(x[0]),int(x[1])),float(x[2])))
    testingrdd = testingrdd.map(lambda x: x.split(',')).filter(lambda x: x[0] != u'userId').map(lambda x: ((int(x[0]),int(x[1])),1))
    trainingrdd = ratingrdd.subtractByKey(testingrdd).map(lambda x: Rating(x[0][0], x[0][1], x[1]))
    verifyrdd = testingrdd.join(ratingrdd).map(lambda x: (x[0],x[1][1]))
    #testing = testingrdd.count()
    #rating = ratingrdd.count()
    #training = trainingrdd.count()
    #verify = verifyrdd.count()
    # print "Training rdd: ", training
    # print "testing rdd: ", testing
    # print "Verifying rdd: ", verify
    model = train_model(trainingrdd)
    predictions = test_model(testingrdd, model)
    #predictions_rdd = s.parallelize(predictions)
    ratesAndPreds = verifyrdd.join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    prediction_baseline = create_baseline(ratesAndPreds).collect()
    with open("Mayank_Kulkarni_ModelBasedCF_Big.txt",'w') as file:
        predictions = sorted(predictions.collect())
        for i in predictions:
            file.write(','.join(str(x) for x in i[0]))
            file.write(","+str(i[1]))
            file.write("\n")
    for i in prediction_baseline:
        if i[0]!=5:
            print ">="+str(i[0]-1)+" and <"+str(i[0])+": "+str(i[1])
        else:
            print ">=" + str(i[0] - 1)+": "+str(i[1])
    print "RMSE: " + str(MSE**0.5)

if __name__ == "__main__":
    t = time.time()
    ratingFileName = sys.argv[1]
    testingFileName = sys.argv[2]

    main(ratingFileName, testingFileName)
    print "Time: ", time.time()-t
