from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from math import radians, sin, cos, sqrt, atan2


# Create a SparkSession
spark = SparkSession.builder \
    .appName("StayLocationCalculator") \
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
df = spark.read.csv("C:/Users/User/Downloads/10.csv", header=True, inferSchema=True)

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

# Define window for stay calculation
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

def update_stays(df):
    window_spec = Window.partitionBy("imsi", "stay_id").orderBy("unix_timestamp")

    return df.withColumn(
        "stay_start_lat", first("latitude").over(window_spec)
    ).withColumn(
        "stay_start_lon", first("longitude").over(window_spec)
    ).withColumn(
        "stay_start_time", first("unix_timestamp").over(window_spec)
    ).withColumn(
        "distance_from_stay_start",
        distance_udf("stay_start_lat", "stay_start_lon", "latitude", "longitude")
    ).withColumn(
        "time_at_location", col("unix_timestamp") - col("stay_start_time")
    ).withColumn(
        "new_stay_change",
        when((col("distance_from_stay_start") > 500) & (col("time_at_location") >= 1200), 1).otherwise(0)
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
                "distance_from_stay_start", "time_at_location", "stay_start_lat", "stay_start_lon").show(25, truncate=False)

# Create a summary of stay locations
stay_locations_summary = df_final.groupBy("imsi", "stay_id").agg(
    first("datetime").alias("start_time"),
    last("datetime").alias("end_time"),
    first("stay_start_lat").alias("stay_latitude"),
    first("stay_start_lon").alias("stay_longitude"),
    (last("unix_timestamp") - first("unix_timestamp")).alias("duration_seconds")
)

# Filter out stays shorter than 20 minutes
stay_locations_summary = stay_locations_summary.filter(col("duration_seconds") >= 1200)

# Show the stay locations summary
stay_locations_summary.orderBy("imsi", "start_time").show(50, truncate=False)

# ---------------------------------------------------------------------------------------------------------------------

from pyspark.sql.functions import when, hour, dayofweek, count, desc, first, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Define function to classify time periods and days
def classify_time_period(hour_col, day_col):
    total_minutes = (hour_col * 60) % 1440  # Convert hour to minutes

    # Weekday patterns (Monday to Friday)
    weekday = day_col.isin([2, 3, 4, 5, 6])
    home_time_weekday = (total_minutes >= 1260) | (total_minutes < 600)  # 9:00 PM to 10:00 AM
    work_time_weekday = (total_minutes >= 660) & (total_minutes < 1200)  # 11:00 AM to 8:00 PM

    # Weekend patterns (Saturday and Sunday)
    weekend = day_col.isin([1, 7])
    weekend_dwell_1 = (total_minutes >= 0) & (total_minutes < 1440)  # 12:00 AM to 12:00 AM
    weekend_dwell_2 = (total_minutes >= 1080) & (total_minutes < 1320)  # 6:00 PM to 10:00 PM

    return when(weekday & home_time_weekday, "Home Time") \
           .when(weekday & work_time_weekday, "Work Time") \
           .when(weekend & weekend_dwell_1, "Weekend Dwell 1") \
           .when(weekend & weekend_dwell_2, "Weekend Dwell 2") \
           .otherwise("Other Time")


# Add hour, day classification, and time period to df_final
df_final = df_final.withColumn("hour", hour("datetime"))
df_final = df_final.withColumn("day_of_week", dayofweek("datetime"))
df_final = df_final.withColumn(
    "time_period",
    classify_time_period(col("hour"), col("day_of_week"))
)

# Aggregate stays by location and time period
aggregated_stays = df_final.filter(col("stay_id") >= 0).groupBy(
    "imsi", "stay_id", "stay_start_lat", "stay_start_lon", "time_period"
).agg(
    count("*").alias("duration_points"),
    (max("unix_timestamp") - min("unix_timestamp")).alias("duration_seconds")
)

# Prepare the data for clustering
vector_assembler = VectorAssembler(inputCols=["stay_start_lat", "stay_start_lon"], outputCol="features")
stays_vector = vector_assembler.transform(aggregated_stays)

# Perform KMeans clustering
num_clusters = 4  # Number of clusters (Home, Work, Club, Other)
kmeans = KMeans().setK(num_clusters).setSeed(1)
model = kmeans.fit(stays_vector)

# Add cluster predictions to the DataFrame
clustered_stays = model.transform(stays_vector)

# Identify the most frequent cluster for each IMSI and time period
frequent_locations = clustered_stays.groupBy("imsi", "prediction", "time_period").agg(
    sum("duration_points").alias("total_duration_points"),
    sum("duration_seconds").alias("total_duration_seconds")
).orderBy("imsi", "time_period", desc("total_duration_seconds"))

# Find the most frequent location for each IMSI and time period
most_frequent_locations = frequent_locations.groupBy("imsi", "time_period").agg(
    first("prediction").alias("most_frequent_cluster"),
    first("total_duration_seconds").alias("duration_seconds")
)

# Join to get the latitude and longitude of the most frequent locations
all_locations = most_frequent_locations.join(
    clustered_stays,
    (most_frequent_locations["imsi"] == clustered_stays["imsi"]) &
    (most_frequent_locations["most_frequent_cluster"] == clustered_stays["prediction"]) &
    (most_frequent_locations["time_period"] == clustered_stays["time_period"])
).select(
    most_frequent_locations["imsi"],
    most_frequent_locations["time_period"],
    most_frequent_locations["most_frequent_cluster"],
    most_frequent_locations["duration_seconds"],
    clustered_stays["stay_start_lat"],
    clustered_stays["stay_start_lon"]
).distinct()

# Aggregate to get the most frequent home, work, and club locations
def get_most_frequent_location(df, time_period):
    return df.filter(col("time_period") == time_period) \
        .groupBy("stay_start_lat", "stay_start_lon") \
        .agg(sum("duration_seconds").alias("total_duration")) \
        .orderBy(desc("total_duration")).first()

home_location = get_most_frequent_location(all_locations, "Home Time")
work_location = get_most_frequent_location(all_locations, "Work Time")
club_location = get_most_frequent_location(all_locations, "Club Time")
other_significant_locations = get_most_frequent_location(all_locations, "Other Time")

# Print the identified locations
def print_location(location_type, location):
    if location:
        print(f"{location_type} Location:")
        print(f"Latitude: {location['stay_start_lat']}, Longitude: {location['stay_start_lon']}")
        print(f"Total duration: {location['total_duration']} seconds")
    else:
        print(f"No {location_type.lower()} location identified.")
    print()

print_location("Home", home_location)
print_location("Work", work_location)
print_location("Club", club_location)
print_location("Other Significant", other_significant_locations)

# Identify potential stops (excluding main locations)
potential_stops = clustered_stays.filter(
    (col("time_period") == "Other Time") &
    (col("duration_seconds") >= 3000)  # At least 50 minutes
).groupBy("imsi", "prediction", "stay_start_lat", "stay_start_lon").agg(
    count("*").alias("stop_frequency"),
    sum("duration_seconds").alias("total_duration")
).orderBy("imsi", desc("total_duration"))

print("Potential stops:")
potential_stops.show()


#---------------------------------------------------------------------
import matplotlib.pyplot as plt

# Collect data from PySpark DataFrame to Pandas DataFrame
sample_size = 10000  # Adjust this value based on your data size and desired plot density
df_sample = df_final.select("latitude", "longitude", "imsi").sample(False, fraction=sample_size/df_final.count())
df_pandas = df_sample.toPandas()

# Create the scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_pandas['longitude'], df_pandas['latitude'],
                      c=df_pandas['imsi'].astype('category').cat.codes,
                      alpha=0.5, s=5)

# Customize the plot
plt.title('Location Data Scatter Plot 10')
plt.xlabel('Longitude')
plt.ylabel('Latitude')


# Add some statistics to the plot
total_points = df_final.count()
unique_imsis = df_final.select("imsi").distinct().count()
plt.text(0.02, 0.98, f'Total points: {total_points}\nUnique IMSIs: {unique_imsis}',
         transform=plt.gca().transAxes, verticalalignment='top')

# Show the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from pyspark.sql.functions import col

# Convert the clustered_locations DataFrame to Pandas
clustered_locations_pd = clustered_locations.toPandas()

# Create a color map for time periods and clusters
color_map = {
    'Home Time': 'red',
    'Work Time': 'blue',
    'Weekend Dwell 1': 'pink',
    'Weekend Dwell 2': 'green',
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
add_significant_location(weekend_location_1, 'Weekend Dwell 1')
add_significant_location(weekend_location_2, 'Weekend Dwell 2')

# Join potential_stops with clustered_locations to get longitude and latitude
potential_stops_with_coords = potential_stops.join(
    clustered_locations.select("imsi", "prediction", "stay_longitude", "stay_latitude").distinct(),
    ["imsi", "prediction"]
)
potential_stops_pd = potential_stops_with_coords.toPandas()

if not potential_stops_pd.empty and 'stay_longitude' in potential_stops_pd.columns and 'stay_latitude' in potential_stops_pd.columns:
    plt.scatter(potential_stops_pd['stay_longitude'], potential_stops_pd['stay_latitude'],
                c='yellow', marker='s', s=50, label='Potential Stops', edgecolors='black')
else:
    print("Warning: No potential stops data available for plotting")

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

#-----------------------------------------------------------------------------------------------------------

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

# Add Weekend activity location 1
if 'weekend_location_1' in locals() and weekend_location_1 is not None:
    folium.Marker(
        location=[weekend_location_1['stay_latitude'], weekend_location_1['stay_longitude']],
        icon=folium.Icon(color='green', icon='leaf'),
        popup=folium.Popup(f"Weekend Activity<br>Latitude: {weekend_location_1['stay_latitude']:.6f}<br>Longitude: {weekend_location_1['stay_longitude']:.6f}", max_width=300)
    ).add_to(m)

# Save the map
map_path = "clustered_locations_visualization_10.html"
m.save(map_path)

# Open the map in the default web browser
webbrowser.open('file://' + os.path.realpath(map_path))

spark.stop()
