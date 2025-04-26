# Stay Location Analysis

This project analyzes user location data to identify significant "stay locations" (places where users spend a considerable amount of time) and attempts to classify them (e.g., Home, Work). It utilizes PySpark for processing large datasets of spatio-temporal points.

## Project Structure

- **`Algorithms/`**: Contains the core PySpark scripts for identifying and analyzing stay locations.
    - **`primary_location/`**: Holds various versions of the main stay location detection algorithm (e.g., `Stay_Location6.py` to `Stay_Location13.py`). These scripts typically identify stays based on a distance threshold (500m). Some versions include additional steps like outlier filtering, time period classification (Home/Work/Weekend), KMeans clustering, and location scoring based on dwell time and frequency.
    - **`Stay_Location_Time.py`**: An alternative algorithm that considers both distance (>500m) and time gaps (>3600s) between location points to define stay boundaries.
- **`Units/`**: Contains utility scripts and potentially earlier/simpler algorithm versions.
    - **`HV_Dist.py`**: Provides a User-Defined Function (UDF) for calculating the Haversine distance between two geographical points.
    - **`Stay_Location_Unit_1.py`, `Stay_Location_Unit_2.py`**: Appear to be basic implementations or components related to the stay location algorithm.
- **`Tests/`**: Includes scripts for generating test data and potentially running validation tests.
    - **`CSV_Location_Generated.py`, `CSV_Location_Generated_2.py`**: Scripts likely used to create synthetic CSV location data for testing the algorithms.
    - **`Stay_Location_Generated.py`, `Stay_Location_Generated_2.py`**: Might be scripts that run the stay location algorithms on generated test data.
- **`Visualizations/`**: Contains HTML output files (e.g., `clustered_locations_visualization_*.html`) that likely visualize the results, such as clustered stay locations on a map.

## Core Logic

1.  **Data Loading**: Reads CSV data containing user identifiers (`imsi`/`user_id`), timestamps, latitude, and longitude.
2.  **Preprocessing**:
    *   Converts date/time strings to Spark timestamps and Unix timestamps.
    *   Calculates time differences and Haversine distances between consecutive points for each user.
    *   Filters data based on coordinate precision and removes outliers based on calculated speed between points.
3.  **Stay Point Identification**:
    *   Identifies potential "stay" boundaries when a user moves more than a specified distance (e.g., 500 meters) from the previous point OR (in `Stay_Location_Time.py`) if the time gap is too large.
    *   Assigns a `stay_id` to consecutive points belonging to the same stay period.
    *   Iteratively refines `stay_id` assignments by checking the distance of each point from the *start* location of its assigned stay.
4.  **Stay Location Summarization**: Groups data by `imsi` and `stay_id` to calculate:
    *   Start and end time of each stay.
    *   Representative latitude/longitude for the stay (usually the first point).
    *   Total dwell time.
5.  **Advanced Analysis (in some scripts)**:
    *   **Time Period Classification**: Categorizes stays into "Home Time", "Work Time", "Weekend Dwell", etc., based on the day of the week and time of day.
    *   **Clustering**: Uses KMeans (k=3) on the stay location coordinates to group stays into significant clusters (potentially Home, Work, Other).
    *   **Scoring**: Ranks locations within each time period/cluster based on a score derived from total dwell time and visit frequency.
    *   **Output**: Prints summary DataFrames and potentially generates visualizations.

## Requirements

-   Python 3.x
-   PySpark
-   A Spark environment (local or cluster)

## Usage

1.  Ensure you have a Spark environment set up.
2.  Modify the input CSV file path within the desired script in the `Algorithms/` directory (e.g., `df = spark.read.csv("YOUR_DATA_PATH.csv", ...)`).
3.  Run the chosen Python script using `spark-submit` or within a PySpark-compatible environment (like a notebook).

   ```bash
   spark-submit Algorithms/primary_location/Stay_Location13.py
   ```

4.  Check the console output for summary tables or the `Visualizations/` folder for generated HTML files (if the script includes visualization steps).

## License

Refer to the `LICENSE` file.
