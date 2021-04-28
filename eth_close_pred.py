import pyspark
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

#Initializing a pyspark session
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
#load .csv file into dataframe
data = spark.read.load("/home/tluan/data/ETHUSD_1hr.csv", format="csv", inferSchema="True", header="true")

featureassembler=VectorAssembler(inputCols=["Open", "High", "Low", "Volume"], outputCol="Independent Features")
output=featureassembler.transform(data)
finalized_data=output.select("Independent Features","Close")
train_data,test_data=finalized_data.randomSplit([0.7,0.3])
regressor=LinearRegression(featuresCol='Independent Features', labelCol='Close')
regressor=regressor.fit(train_data)

pred_results=regressor.evaluate(test_data)

trainingSummary = regressor.summary

print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)