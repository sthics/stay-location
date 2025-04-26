import csv
from datetime import datetime, timedelta
import random
import math


def generate_point(center_lat, center_lon, radius):
    r = radius * math.sqrt(random.random())
    theta = random.random() * 2 * math.pi

    dx = r * math.cos(theta)
    dy = r * math.sin(theta)

    lat = center_lat + (dy / 111111)
    lon = center_lon + (dx / (111111 * math.cos(math.radians(center_lat))))

    return lat, lon


def generate_location_data(start_time, end_time, center_lat, center_lon):
    data = []
    current_time = start_time
    while current_time < end_time:
        lat, lon = generate_point(center_lat, center_lon, 100)
        end = min(current_time + timedelta(minutes=10), end_time)
        data.append([lat, lon, current_time.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")])
        current_time = end
    return data


def home_location(start, end):
    return generate_location_data(start, end, 21.14524192130262, 79.06461660731819)  # Apartments


def work_location(start, end):
    return generate_location_data(start, end, 21.084383948394, 79.08459484934)  # Location Guru


def gym_location(start, end):
    return generate_location_data(start, end, 21.139207966009433, 79.06013432365904)  # RELOAD fitness gym


def random_location(start, end):
    return generate_location_data(start, end, 21.151523574769946, 79.0659592620449)  # Bella Brew


def travel(start_coords, end_coords, distance, start_time):
    duration = timedelta(hours=1) if distance == 6 else timedelta(minutes=20)
    data = []
    current_time = start_time
    step = duration / 6  # 6 points for travel
    for i in range(6):
        lat = start_coords[0] + (end_coords[0] - start_coords[0]) * (i / 5)
        lon = start_coords[1] + (end_coords[1] - start_coords[1]) * (i / 5)
        data.append(
            [lat, lon, current_time.strftime("%Y-%m-%d %H:%M:%S"), (current_time + step).strftime("%Y-%m-%d %H:%M:%S")])
        current_time += step
    return data


def generate_data():
    start_date = datetime(2024, 1, 1)  # Starting from January 1, 2024
    data = []
    current_time = start_date.replace(hour=0, minute=0, second=0)

    for i in range(31):  # One month
        # Home (from midnight or previous day's end, until leaving for work)
        work_start = current_time.replace(hour=random.randint(9, 10), minute=random.randint(0, 59), second=0)
        home_data = home_location(current_time, work_start)
        data.extend(home_data)
        current_time = work_start

        # Travel to work
        work_travel = travel((home_data[-1][0], home_data[-1][1]), (21.084383948394, 79.08459484934), 6, current_time)
        data.extend(work_travel)
        current_time = datetime.strptime(work_travel[-1][3], "%Y-%m-%d %H:%M:%S")

        # Work
        work_end = current_time.replace(hour=random.randint(19, 20), minute=random.randint(0, 30), second=0)
        work_data = work_location(current_time, work_end)
        data.extend(work_data)
        current_time = work_end

        # Travel to gym
        gym_travel = travel((work_data[-1][0], work_data[-1][1]), (21.139207966009433, 79.06013432365904), 3,
                            current_time)
        data.extend(gym_travel)
        current_time = datetime.strptime(gym_travel[-1][3], "%Y-%m-%d %H:%M:%S")

        # Gym
        gym_duration = timedelta(minutes=random.randint(30, 90))
        gym_end = current_time + gym_duration
        gym_data = gym_location(current_time, gym_end)
        data.extend(gym_data)
        current_time = gym_end

        # Travel back home
        home_travel = travel((gym_data[-1][0], gym_data[-1][1]), (21.14524192130262, 79.06461660731819), 3,
                             current_time)
        data.extend(home_travel)
        current_time = datetime.strptime(home_travel[-1][3], "%Y-%m-%d %H:%M:%S")

        # Random location (not every day)
        if random.random() < 0.4:  # 40% chance
            random_start = current_time.replace(hour=random.randint(22, 22), minute=random.randint(0, 59), second=0)
            if random_start > current_time:
                home_data_evening = home_location(current_time, random_start)
                data.extend(home_data_evening)
                current_time = random_start

            random_duration = timedelta(minutes=random.randint(30, 120))
            random_end = random_start + random_duration
            random_data = random_location(random_start, random_end)
            data.extend(random_data)
            current_time = random_end

        # Ensure we have data until midnight
        next_day = (current_time + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        if current_time < next_day:
            final_home_data = home_location(current_time, next_day)
            data.extend(final_home_data)

        current_time = next_day

    return data


# Generate data
one_month_data = generate_data()

# Write to CSV
with open('person_location_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['latitude', 'longitude', 'starttime', 'endtime'])
    writer.writerows(one_month_data)

print("CSV file 'person_location_data.csv' has been generated.")