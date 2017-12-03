# -*- coding: utf-8 -*-  
from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

# 创建sc环境
sc = SparkContext()
sqlContext = SQLContext(sc)

# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, 'data/mllib/sample_libsvm_data.txt') # 数据默认存放在 SPARK_HOME 目录下

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3]) # 随机分隔数据

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = (labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())) 

print "Test Error = %2.4f%%" % (float(testErr)*100)
print "Learned classification tree model:"
print model.toDebugString()


# Test Error = 3.85%  took 0.103933 s

# Save and load model
#model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
#sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")