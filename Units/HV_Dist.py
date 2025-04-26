import pyspark
import math

from math import radians, cos, sin, asin, sqrt
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r

# Register the function as a UDF (User-Defined Function)
haversine_udf = udf(haversine_distance, DoubleType())

# Example usage

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DistanceCalculator").getOrCreate()

data = [
    (21.1005340601974,  79.0938842036432, 21.128414877063715, 73.04208507115538),
    (21.140973257135773, 79.07959315606163, 21.1005340601974, 79.0938842036432)
]

df = spark.createDataFrame(data, ["lat1", "lon1", "lat2", "lon2"])
df = df.withColumn("distance", haversine_udf("lat1", "lon1", "lat2", "lon2"))

df.show()