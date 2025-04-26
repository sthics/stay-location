from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from math import radians, sin, cos, sqrt, atan2


# Create a SparkSession
spark = SparkSession.builder \
    .appName("StayLocation13") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

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

# Read the CSV file into a DataFrame
df = spark.read.csv("C:/Users/User/Downloads/13.csv", header=True, inferSchema=True)

# Adjust date and start_time to be in the same format
df = df.withColumn(
    "datetime",
    when(
        concat(col("date"), lit(" "), col("start_time")).rlike("(AM|PM)"),
        from_unixtime(
            unix_timestamp(
                concat(substring(col("date"), 1, 10), lit(" "), substring(col("start_time"), 1, 11)),
                "MM/dd/yyyy hh:mm:ss a"
            )
        )
    ).otherwise(
        from_unixtime(
            unix_timestamp(
                concat(substring(col("date"), 1, 10), lit(" "), substring(col("start_time"), 1, 8)),
                "MM/dd/yyyy HH:mm:ss"
            )
        )
    )
)

# Convert datetime string to timestamp and then to Unix timestamp (seconds)
df = df.withColumn("timestamp", to_timestamp("datetime"))
df = df.withColumn("unix_timestamp", unix_timestamp("timestamp"))

# Define window specifications
user_window = Window.partitionBy("imsi").orderBy("unix_timestamp")

# Drop location column
df = df.drop(col("location"))


# Rename location.lat and location.lon
df = df.withColumnRenamed("location.lat", "latitude") \
       .withColumnRenamed("location.lon", "longitude")

# Filter out missing values
df = df.filter(col("latitude").isNotNull() & col("longitude").isNotNull())

from pyspark.sql.functions import regexp_extract

# Define regular expression patterns for lat/lon with at least 5 decimal places
lat_lon_pattern = r'-?\d+\.\d{5,}'

# Apply the filter
df_filtered1 = df.filter(
    (length(regexp_extract(col("latitude").cast("string"), lat_lon_pattern, 0)) > 0) &
    (length(regexp_extract(col("longitude").cast("string"), lat_lon_pattern, 0)) > 0)
)

# Use df_filtered for further processing
df = df_filtered1

from pyspark.sql.functions import lag, lead

# Define a reasonable speed threshold (e.g., 200 km/h or about 55.6 m/s)
speed_threshold = 60  # meters per second

# Calculate speed between consecutive points
df_with_speed = df.withColumn(
    "next_lat", lead("latitude").over(user_window)
).withColumn(
    "next_lon", lead("longitude").over(user_window)
).withColumn(
    "next_timestamp", lead("unix_timestamp").over(user_window)
).withColumn(
    "distance_to_next",
    distance_udf("latitude", "longitude", "next_lat", "next_lon")
).withColumn(
    "time_diff", col("next_timestamp") - col("unix_timestamp")
).withColumn(
    "speed", when(col("time_diff") > 0, col("distance_to_next") / col("time_diff")).otherwise(0)
)

# Filter out or flag outliers
df_filtered = df_with_speed.withColumn(
    "is_outlier", (col("speed") > speed_threshold) | (col("speed").isNull())
)

# Remove outliers
df_without_outliers = df_filtered.filter(~col("is_outlier"))
df = df_without_outliers

#User window
user_window = Window.partitionBy("imsi").orderBy("unix_timestamp")

# Calculate distance from previous point
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
    window_spec = Window.partitionBy("imsi", "stay_id").orderBy("unix_timestamp")

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
df_final.select("datetime", "imsi", "latitude", "longitude", "stay_id", "distance_from_prev",
                "distance_from_stay_start", "stay_start_lat", "stay_start_lon").show(25, truncate=False)

# Create a summary of stay locations
stay_locations_summary = df_final.groupBy("imsi", "stay_id").agg(
    first("datetime").alias("start_time"),
    last("datetime").alias("end_time"),
    first("stay_start_lat").alias("stay_latitude"),
    first("stay_start_lon").alias("stay_longitude")
)

# Show the stay locations summary
stay_locations_summary.orderBy("imsi", "start_time").show(25, truncate=False)


# ----------------------------------------------------------------------------------------------------------------------
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.window import Window

def classify_time_period(hour_col, day_col):
    total_minutes = (hour_col * 60) % 1440  # Convert hour to minutes

    # Weekday patterns
    weekday = day_col.isin([2, 3, 4, 5, 6])  # Monday to Friday
    home_time_weekday = (total_minutes >= 960) | (total_minutes < 660)  # 4:00 PM to 10:00 AM
    work_time_weekday = (total_minutes >= 660) & (total_minutes < 900)  # 11:00 AM to 3:00 PM

    # Weekend pattern (Sunday)
    sunday = (day_col == 1)
    weekend_dwell = (total_minutes >= 420) & (total_minutes < 720)  # 7:00 AM to 12:00 PM

    return when(weekday & home_time_weekday, "Home Time") \
           .when(weekday & work_time_weekday, "Work Time") \
           .when(sunday & weekend_dwell, "Weekend Dwell") \
           .otherwise("Other Time")

# Calculate dwell time in hours
stay_locations_summary = stay_locations_summary.withColumn(
    "dwell_time_hours",
    (unix_timestamp("end_time") - unix_timestamp("start_time")) / 3600
)

# Add hour and day classification to the DataFrame
stay_locations_summary = stay_locations_summary.withColumn("hour", hour("start_time"))
stay_locations_summary = stay_locations_summary.withColumn("day_of_week", dayofweek("start_time"))

stay_locations_summary = stay_locations_summary.withColumn(
    "time_period",
    classify_time_period(col("hour"), col("day_of_week"))
)

# Prepare the data for clustering
vector_assembler = VectorAssembler(inputCols=["stay_latitude", "stay_longitude"], outputCol="features")
stay_locations_vector = vector_assembler.transform(stay_locations_summary)

# Perform KMeans clustering
num_clusters = 3  # Number of clusters (Home, Work, Other)
kmeans = KMeans().setK(num_clusters).setSeed(1)
model = kmeans.fit(stay_locations_vector)

# Add cluster predictions to the DataFrame
clustered_locations = model.transform(stay_locations_vector)

# Calculate total dwell time and visit frequency for each location and time period
location_metrics = clustered_locations.groupBy("imsi", "prediction", "time_period", "stay_latitude", "stay_longitude") \
    .agg(
        sum("dwell_time_hours").alias("total_dwell_time"),
        count("*").alias("visit_frequency")
    )

# Score locations based on dwell time and visit frequency
location_metrics = location_metrics.withColumn(
    "score",
    col("total_dwell_time") * col("visit_frequency")
)

# Find the location with the highest score for each time period
window_spec = Window.partitionBy("imsi", "time_period").orderBy(desc("score"))
top_locations = location_metrics.withColumn("rank", row_number().over(window_spec)) \
    .filter(col("rank") == 1) \
    .drop("rank")

# Identify home, work, and weekend locations
home_location = top_locations.filter(col("time_period") == "Home Time") \
    .orderBy(desc("score")).first()

work_location = top_locations.filter(col("time_period") == "Work Time") \
    .orderBy(desc("score")).first()

weekend_location = top_locations.filter(col("time_period") == "Weekend Dwell") \
    .orderBy(desc("score")).first()

other_significant_locations = top_locations.filter(col("time_period") == "Other Time") \
    .orderBy(desc("score")).collect()

def print_location(location_type, location):
    if location:
        print(f"{location_type} Location:")
        print(f"Latitude: {location['stay_latitude']}, Longitude: {location['stay_longitude']}")
        print(f"Total dwell time: {location['total_dwell_time']:.2f} hours")
        print(f"Visit frequency: {location['visit_frequency']}")
        print(f"Score: {location['score']:.2f}")
    else:
        print(f"No {location_type.lower()} location identified.")
    print()

# Print the identified locations
print_location("Home", home_location)
print_location("Work", work_location)
print_location("Weekend Dwell", weekend_location)

print("Other Significant Locations:")
for location in other_significant_locations:
    print_location("Other", location)

# Identify potential stops (excluding main locations)
potential_stops = clustered_locations.filter(
    (col("time_period") == "Other Time") &
    (col("dwell_time_hours") >= 0.67)  # At least 40 minutes (travel time)
).groupBy("imsi", "prediction", "stay_latitude", "stay_longitude").agg(
    count("*").alias("stop_frequency"),
    sum("dwell_time_hours").alias("total_dwell_time")
).orderBy("imsi", desc("stop_frequency"))

print("\nPotential stops:")
potential_stops.show()

# ----------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Define sample size
sample_size = 10000  # Adjust this value based on your data size and desired plot density

# Calculate the total number of rows in the DataFrame
total_rows = df_final.count()

# Determine the sampling fraction
if sample_size > total_rows:
    sample_fraction = 1.0  # Use the whole DataFrame if the sample size is greater than the total number of rows
else:
    sample_fraction = sample_size / total_rows

# Collect a sample from the DataFrame
df_sample = df_final.select("latitude", "longitude", "imsi").sample(False, sample_fraction)
df_pandas = df_sample.toPandas()

# Create the scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_pandas['longitude'], df_pandas['latitude'],
                      c=df_pandas['imsi'].astype('category').cat.codes,
                      alpha=0.5, s=5)

# Customize the plot
plt.title('Location Data Scatter Plot 13')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Optional: Add some statistics to the plot
plt.text(0.02, 0.98, f'Total points: {total_rows}\nUnique IMSIs: {df_final.select("imsi").distinct().count()}',
         transform=plt.gca().transAxes, verticalalignment='top')

# Show the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Convert the clustered_locations DataFrame to Pandas
clustered_locations_pd = clustered_locations.toPandas()

# Create a color map for time periods and clusters
color_map = {
    'Home Time': 'red',
    'Work Time': 'blue',
    'Other Time': 'gray'
}

# Create the scatter plot
plt.figure(figsize=(12, 8))

# Plot each cluster
for time_period, color in color_map.items():
    cluster_data = clustered_locations_pd[clustered_locations_pd['time_period'] == time_period]
    plt.scatter(cluster_data['stay_longitude'], cluster_data['stay_latitude'],
                c=color, label=time_period, alpha=0.6, s=10)

# Add significant locations
def add_significant_location(location, name):
    if location:
        plt.scatter(location['stay_longitude'], location['stay_latitude'],
                    c=color_map.get(name, 'black'), marker='*', s=200, label=f'{name} (Main)', edgecolors='black')

add_significant_location(home_location, 'Home Time')
add_significant_location(work_location, 'Work Time')

# Customize the plot
plt.title('Location Data Scatter Plot (KMeans Clustering Results)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

# Add some statistics to the plot
total_points = clustered_locations_pd.shape[0]
unique_clusters = clustered_locations_pd['prediction'].nunique()
plt.text(0.02, 0.98, f'Total points: {total_points}\nUnique clusters: {unique_clusters}',
         transform=plt.gca().transAxes, verticalalignment='top')

# Show the plot
plt.tight_layout()
plt.show()

import folium
from folium.plugins import MarkerCluster
from branca.colormap import LinearColormap
import webbrowser
import os

# Create a map centered on the mean coordinates
m = folium.Map(location=[clustered_locations_pd['stay_latitude'].mean(),
                         clustered_locations_pd['stay_longitude'].mean()],
               zoom_start=10)

# Create a colormap for the clusters
colormap = LinearColormap(colors=['purple', 'blue', 'green', 'yellow', 'orange', 'red'],
                          vmin=clustered_locations_pd['prediction'].min(),
                          vmax=clustered_locations_pd['prediction'].max())

# Add markers for all points
marker_cluster = MarkerCluster().add_to(m)

for idx, row in clustered_locations_pd.iterrows():
    folium.CircleMarker(
        location=[row['stay_latitude'], row['stay_longitude']],
        radius=5,
        popup=f"Time Period: {row['time_period']}",
        color=colormap(row['prediction']),
        fill=True,
        fillColor=colormap(row['prediction']),
        fillOpacity=0.7
    ).add_to(marker_cluster)

# Add colormap to the map
colormap.add_to(m)
colormap.caption = 'Cluster Prediction'

# Add Home location with coordinates in popup
if 'home_location' in locals() and home_location is not None:
    folium.Marker(
        location=[home_location['stay_latitude'], home_location['stay_longitude']],
        icon=folium.Icon(color='red', icon='home'),
        popup=folium.Popup(f"Home<br>Latitude: {home_location['stay_latitude']:.6f}<br>Longitude: {home_location['stay_longitude']:.6f}", max_width=300)
    ).add_to(m)

# Add Work location
if 'work_location' in locals() and work_location is not None:
    folium.Marker(
        location=[work_location['stay_latitude'], work_location['stay_longitude']],
        icon=folium.Icon(color='blue', icon='briefcase'),
        popup=folium.Popup(f"Work<br>Latitude: {work_location['stay_latitude']:.6f}<br>Longitude: {work_location['stay_longitude']:.6f}", max_width=300)
    ).add_to(m)


# Save the map
map_path = "clustered_locations_visualization_13.html"
m.save(map_path)

# Open the map in the default web browser
webbrowser.open('file://' + os.path.realpath(map_path))

spark.stop()


#-----------------------------------------------------------------------------------------------------------------------

