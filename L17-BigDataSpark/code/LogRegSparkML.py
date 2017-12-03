# -*- coding: utf-8 -*-  
from pyspark.ml.classification import LogisticRegression

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("LogisticRegressionSparkML").getOrCreate()

# Load training data
training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for logistic regression
print "Coefficients: %s" % str(lrModel.coefficients)
print "Intercept: %s" % str(lrModel.intercept)

# We can also use the multinomial family for binary classification
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

# Fit the model
mlrModel = mlr.fit(training)

# Print the coefficients and intercepts for logistic regression with multinomial family
print "Multinomial coefficients: %s" % str(mlrModel.coefficientMatrix)
print "Multinomial intercepts: %s" % str(mlrModel.interceptVector)

spark.stop() 