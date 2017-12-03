# -*- coding: utf-8 -*-  
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("NBSparkMLlib").getOrCreate()

# Load training data
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")# 数据默认存放在 SPARK_HOME 目录下

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print "Test set accuracy = %2.4f%%" % (float(accuracy) * 100)

spark.stop() 