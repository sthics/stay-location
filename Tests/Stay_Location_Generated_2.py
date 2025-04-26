from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, lag, when, sum, first, last, to_timestamp, unix_timestamp, date_trunc
from pyspark.sql.types import DoubleType
from math import radians, sin, cos, sqrt, atan2

# Haversine distance function
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

# Register UDF
distance_udf = spark.udf.register("distance_udf", haversine_distance, DoubleType())

# Read the CSV file into a DataFrame
df = spark.read.csv("geographic_points.csv", header=True, inferSchema=True)

# Convert datetime to timestamp and extract date
df = df.withColumn("timestamp", to_timestamp("datetime"))
df = df.withColumn("date", date_trunc("day", col("timestamp")))
df = df.withColumn("unix_timestamp", unix_timestamp("timestamp"))

# Define window specifications with partitioning by date
window = Window.partitionBy("date").orderBy("unix_timestamp")

# Calculate distance from previous point
df = df.withColumn("prev_lat", lag("latitude").over(window))
df = df.withColumn("prev_lon", lag("longitude").over(window))
df = df.withColumn("distance_from_prev",
    distance_udf("prev_lat", "prev_lon", "latitude", "longitude")
)

# Initialize stay changes
df = df.withColumn("stay_change", when(col("distance_from_prev") > 500, 1).otherwise(0))

# Assign initial stay IDs
df = df.withColumn("stay_id", sum("stay_change").over(window))

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

    # Update stay_id only when the distance from the start exceeds 500 meters
    df = df.withColumn(
        "new_stay_id",
        when(col("distance_from_stay_start") > 500, col("stay_id") + 1).otherwise(col("stay_id"))
    )

    # Recalculate stay_start_lat and stay_start_lon based on new_stay_id
    window_spec_new = Window.partitionBy("date", "new_stay_id").orderBy("unix_timestamp")
    df = df.withColumn(
        "stay_start_lat", first("latitude").over(window_spec_new)
    ).withColumn(
        "stay_start_lon", first("longitude").over(window_spec_new)
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
df_final.select("datetime", "latitude", "longitude", "stay_id", "distance_from_prev",
                "distance_from_stay_start", "stay_start_lat", "stay_start_lon").show(truncate=False)

# Create a summary of stay locations
stay_locations_summary = df_final.groupBy("date", "stay_id").agg(
    first("datetime").alias("start_time"),
    last("datetime").alias("end_time"),
    first("stay_start_lat").alias("stay_latitude"),
    first("stay_start_lon").alias("stay_longitude")
)

# Show the stay locations summary
stay_locations_summary.orderBy("date", "start_time").show(truncate=False)

spark.stop()
