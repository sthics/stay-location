from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, lag, when, sum, first, last, to_timestamp, unix_timestamp, date_trunc
from pyspark.sql.types import DoubleType
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    R = 6371000  # Earth's radius in meters
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Create SparkSession
spark = SparkSession.builder \
    .appName("StayLocationCalculator") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

distance_udf = spark.udf.register("distance_udf", haversine_distance, DoubleType())

# Read the CSV file into a DataFrame
df = spark.read.csv("person_location_data.csv", header=True, inferSchema=True)

# Convert start time to timestamp and extract date
df = df.withColumn("timestamp", to_timestamp("starttime"))
df = df.withColumn("date", date_trunc("day", col("timestamp")))
df = df.withColumn("unix_timestamp", unix_timestamp("timestamp"))

# Define window specifications
user_window = Window.partitionBy("date").orderBy("unix_timestamp")

# Calculate distance from previous point
df = df.withColumn("prev_lat", lag("latitude").over(user_window))
df = df.withColumn("prev_lon", lag("longitude").over(user_window))
df = df.withColumn("distance_from_prev",
    distance_udf("prev_lat", "prev_lon", "latitude", "longitude")
)

# Initialize stay changes
df = df.withColumn(
    "stay_change", when(col("distance_from_prev") > 500, 1).otherwise(0)
)

# Assign initial stay IDs
df = df.withColumn(
    "stay_id", sum("stay_change").over(user_window)
)

# Function to update stay IDs
def update_stays(df):
    window_spec = Window.partitionBy("date", "stay_id").orderBy("unix_timestamp")
    df = df.withColumn(
        "stay_start_lat", first("latitude").over(window_spec)
    ).withColumn(
        "stay_start_lon", first("longitude").over(window_spec)
    )
    df = df.withColumn(
        "distance_from_stay_start",
        distance_udf("stay_start_lat", "stay_start_lon", "latitude", "longitude")
    )
    df = df.withColumn(
        "new_stay_change", when(col("distance_from_stay_start") > 500, 1).otherwise(0)
    ).withColumn(
        "new_stay_id",
        sum("new_stay_change").over(user_window) + col("stay_id")
    )
    return df

# Apply the function iteratively
for i in range(3):  # Adjust the number of iterations if needed
    df = update_stays(df)
    if i < 2:  # Don't rename on the last iteration
        df = df.drop("stay_id").withColumnRenamed("new_stay_id", "stay_id")

# Final update to get the correct stay start coordinates
df_final = df.drop("stay_id").withColumnRenamed("new_stay_id", "stay_id")

# Show the results
df_final.select("starttime", "date", "latitude", "longitude", "stay_id", "distance_from_prev",
                "distance_from_stay_start", "stay_start_lat", "stay_start_lon").show(25, truncate=False)

# Create a summary of stay locations
stay_locations_summary = df_final.groupBy("date", "stay_id").agg(
    first("starttime").alias("start_time"),
    last("endtime").alias("end_time"),
    first("stay_start_lat").alias("stay_latitude"),
    first("stay_start_lon").alias("stay_longitude")
)

# Show the stay locations summary
stay_locations_summary.orderBy("date", "start_time").show(100, truncate=False)

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Prepare the data for clustering
vector_assembler = VectorAssembler(inputCols=["stay_latitude", "stay_longitude"], outputCol="features")
stay_locations_vector = vector_assembler.transform(stay_locations_summary)

# Perform KMeans clustering
num_clusters = 3  # Number of clusters
kmeans = KMeans().setK(num_clusters).setSeed(1)
model = kmeans.fit(stay_locations_vector)

# Add cluster predictions to the DataFrame
clustered_locations = model.transform(stay_locations_vector)

# Evaluate the clusters
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(clustered_locations)
print(f"Silhouette with squared euclidean distance = {silhouette}")

# Show the clustered results
clustered_locations.select( "date","start_time", "end_time", "stay_latitude", "stay_longitude", "prediction").show()

# Get cluster centers
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Identify the most frequent cluster for each date (potential home/work locations)
from pyspark.sql.functions import count, desc

frequent_locations = clustered_locations.groupBy("date", "prediction").agg(count("*").alias("frequency")) \
    .orderBy("date", desc("frequency"))

frequent_locations.show()

"""
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import count, desc, first

# Prepare the data for clustering
vector_assembler = VectorAssembler(inputCols=["stay_latitude", "stay_longitude"], outputCol="features")
stay_locations_vector = vector_assembler.transform(stay_locations_summary)

# Perform Gaussian Mixture Model clustering
num_clusters = 4  # Number of clusters
gmm = GaussianMixture().setK(num_clusters).setSeed(1)
model = gmm.fit(stay_locations_vector)

# Add cluster predictions to the DataFrame
clustered_locations = model.transform(stay_locations_vector)

# Evaluate the clusters
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(clustered_locations)
print(f"Silhouette with squared euclidean distance = {silhouette}")

# Show the clustered results
clustered_locations.select("date", "start_time", "end_time", "stay_latitude", "stay_longitude", "prediction").show()

# Get cluster parameters
gaussians = model.gaussiansDF
print("Gaussian distributions for each cluster:")
gaussians.show(truncate=False)

# Identify the most frequent cluster for each date (potential home/work locations)
frequent_locations = clustered_locations.groupBy("date", "prediction").agg(count("*").alias("frequency")) \
    .orderBy("date", desc("frequency"))

# Find the most frequent location for each date
most_frequent_locations = frequent_locations.groupBy("date").agg(first("prediction").alias("most_frequent_cluster"))

# Join to get the latitude and longitude of the most frequent locations
home_work_locations = most_frequent_locations.join(
    clustered_locations,
    (most_frequent_locations["date"] == clustered_locations["date"]) &
    (most_frequent_locations["most_frequent_cluster"] == clustered_locations["prediction"])
).select(
    most_frequent_locations["date"],
    most_frequent_locations["most_frequent_cluster"],
    clustered_locations["stay_latitude"],
    clustered_locations["stay_longitude"]
).distinct()

# Print the frequent locations
home_work_locations.show()

# Print cluster weights
weights = model.weights
print("Weights of each Gaussian component:")
for i, weight in enumerate(weights):
    print(f"Cluster {i}: {weight}")

# Stop SparkSession
spark.stop()
"""