# -*- coding: utf-8 -*-  

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.fpm import FPGrowth

# 创建sc环境
sc = SparkContext()
sqlContext = SQLContext(sc)

data = sc.textFile("/home/lanman/course/dm/data/sample_fpgrowth.txt")
transactions = data.map(lambda line: line.strip().split(' '))
model = FPGrowth.train(transactions, minSupport=0.2, numPartitions=10)
result = model.freqItemsets().collect()

for fi in result:
	print fi

# 输出结果到文本文件
resultfile = open('/home/lanman/course/dm/data/output-fpgrwothSparkMLlibResult.txt','w')
for fi in result:
    print(fi)
    print >> resultfile, fi 

resultfile.close()