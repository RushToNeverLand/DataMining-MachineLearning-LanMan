# -*- coding: utf-8 -*-  

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split('\t')]
    return LabeledPoint(int(values[-1]), values[:-1])  # ��һ�������class label���ڶ�����features

# ����sc����
sc = SparkContext()
sqlContext = SQLContext(sc)

# ����training data
trainingData = sc.textFile("/home/lanman/course/dm/data/horseColicTraining.txt") # ʹ�þ���·��������$SPARK_HOMEĿ¼��·��
trainingParsedData = trainingData.map(parsePoint)

# Build the model on Training data
model = SVMWithSGD.train(trainingParsedData, iterations=100)

# ����test data
testData = sc.textFile("/home/lanman/course/dm/data/horseColicTest.txt") # ʹ�þ���·��������$SPARK_HOMEĿ¼��·��
testParsedData = testData.map(parsePoint)

# Evaluating the model on Test data
labelsAndPreds = testParsedData.map(lambda p: (p.label, model.predict(p.features)))
testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(testParsedData.count())

print "Test Error = %2.4f%%" % (float(testErr)*100)

# LogReg�Ľ��
#Training Error = 27.7592%
#Test Error = 28.3582%

# SVM�Ľ����Ĭ�ϲ���
#Test Error = 50.7463% ��iterations=100��took 0.075601 s
#Test Error = 49.2537%��iterations=300��

# Save and load model
#model.save(sc, "/home/lanman/course/dm/model/mySVMModelPath")
#sameModel = LogisticRegressionModel.load(sc, "/home/lanman/course/dm/model/mySVMModelPath")
