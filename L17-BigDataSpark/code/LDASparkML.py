# -*- coding: utf-8 -*-  
from pyspark.ml.clustering import LDA

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("LDASparkML").getOrCreate()

# Loads data.
dataset = spark.read.format("libsvm").load("data/mllib/sample_lda_libsvm_data.txt")

# Trains a LDA model.
lda = LDA(k=10, maxIter=10)
model = lda.fit(dataset)

ll = model.logLikelihood(dataset)
lp = model.logPerplexity(dataset)
print "The lower bound on the log likelihood of the entire corpus: %s" % str(ll)
print "The upper bound bound on perplexity: %s" % str(lp)

# Describe topics.
topics = model.describeTopics(3)
print "The topics described by their top-weighted terms:"
topics.show(truncate=False)

# Shows the result
transformed = model.transform(dataset)
transformed.show(truncate=False)

spark.stop() 