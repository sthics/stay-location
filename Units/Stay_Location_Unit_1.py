from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from math import radians, sin, cos, sqrt, atan2

# Create a SparkSession
spark = SparkSession.builder.appName("StayLocationCalculator").getOrCreate()

# Define a UDF to calculate distance between two lat-long pairs
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

distance_udf = udf(haversine_distance)


# Assume we have a DataFrame 'df' with columns: datetime, user_id, latitude, longitude
df = spark.read.csv("C:/Path/to/csv", header=True, inferSchema=True)

# Convert datetime string to timestamp and then to Unix timestamp (seconds)
df = df.withColumn("timestamp", to_timestamp("datetime"))
df = df.withColumn("unix_timestamp", unix_timestamp("timestamp"))

# Define window specifications
user_window = Window.partitionBy("user_id").orderBy("unix_timestamp")

df_with_distance = df.withColumn(
    "prev_lat", lag("latitude").over(user_window)
).withColumn(
    "prev_lon", lag("longitude").over(user_window)
).withColumn(
    "distance_from_prev",
    distance_udf("prev_lat", "prev_lon", "latitude", "longitude")
)

# Initialize stay changes
df_with_stay_init = df_with_distance.withColumn(
    "stay_change", when(col("distance_from_prev") > 500, 1).otherwise(0)
)

# Assign initial stay IDs
df_with_stay_id = df_with_stay_init.withColumn(
    "stay_id", sum("stay_change").over(user_window)
)


# Function to update stay IDs
def update_stays(df):
    window_spec = Window.partitionBy("user_id", "stay_id").orderBy("unix_timestamp")

    return df.withColumn(
        "stay_start_lat", first("latitude").over(window_spec)
    ).withColumn(
        "stay_start_lon", first("longitude").over(window_spec)
    ).withColumn(
        "distance_from_stay_start",
        distance_udf("stay_start_lat", "stay_start_lon", "latitude", "longitude")
    ).withColumn(
        "new_stay_change", when(col("distance_from_stay_start") > 500, 1).otherwise(0)
    ).withColumn(
        "new_stay_id",
        sum("new_stay_change").over(user_window) + col("stay_id")
    )


# Apply the function iteratively to handle cases where multiple changes might be needed
for i in range(3):  # Adjust the number of iterations if needed
    df_with_stay_id = update_stays(df_with_stay_id)
    if i < 2:  # Don't rename on the last iteration
        df_with_stay_id = df_with_stay_id.drop("stay_id").withColumnRenamed("new_stay_id", "stay_id")

# Final update to get the correct stay start coordinates
df_final = df_with_stay_id.drop("stay_id").withColumnRenamed("new_stay_id", "stay_id")

# Show the results
df_final.select("datetime", "user_id", "latitude", "longitude", "stay_id", "distance_from_prev",
                "distance_from_stay_start", "stay_start_lat", "stay_start_lon").show(25, truncate=False)

# Create a summary of stay locations
stay_locations_summary = df_final.groupBy("user_id", "stay_id").agg(
    first("datetime").alias("start_time"),
    last("datetime").alias("end_time"),
    first("stay_start_lat").alias("stay_latitude"),
    first("stay_start_lon").alias("stay_longitude")
)

# Show the stay locations summary
stay_locations_summary.orderBy("user_id", "start_time").show(25, truncate=False)