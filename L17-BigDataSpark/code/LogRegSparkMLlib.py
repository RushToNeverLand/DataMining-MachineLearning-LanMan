# -*- coding: utf-8 -*-  

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split('\t')]  # 特征以tab分隔，最后一列是类标
    return LabeledPoint(int(values[-1]), values[:-1])  # 第一个是类标class label，第二个是features

# 创建sc环境
sc = SparkContext()
sqlContext = SQLContext(sc)

# 解析training data
trainingData = sc.textFile("/home/lanman/course/dm/data/horseColicTraining.txt") # 使用绝对路径，或者$SPARK_HOME目录下路径
trainingParsedData = trainingData.map(parsePoint)

# Build the model on Training data
model = LogisticRegressionWithLBFGS.train(trainingParsedData, iterations=300)

# 解析test data
testData = sc.textFile("/home/lanman/course/dm/data/horseColicTest.txt") # 使用绝对路径，或者$SPARK_HOME目录下路径
testParsedData = testData.map(parsePoint)

# Evaluating the model on Test data
labelsAndPreds = testParsedData.map(lambda p: (p.label, model.predict(p.features)))
testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(testParsedData.count())

print "Test Error = %2.4f%%" % (float(testErr)*100)

#Test Error = 28.3582%  took 0.097996 s (iterations=100)
#Test Error = 28.3582%  took 0.067363 s (iterations=300)

# Save and load model
#model.save(sc, "/home/lanman/course/dm/model/myLogRegSparkModelPath")
#sameModel = LogisticRegressionModel.load(sc, "/home/lanman/course/dm/model/myLogRegSparkModelPath")