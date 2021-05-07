import calendar
import time
import math

OUTPUT_FILENAME = "activity.txt"

LOG_FILENAME = "fait-taxi-failed.log"
DATASET_NAME = "waymo"
TRAINING_THRESHOLD = 20  # seconds
PAUSE_THRESHOLD = 120  # seconds

lines = []
with open(LOG_FILENAME, "r") as f:
    for line in f:
        parts = line.strip().split(" - ")
        timestamp = parts[0]
        timestamp = time.strptime(timestamp[: timestamp.find(",")], "%Y-%m-%d %H:%M:%S")
        timestamp = calendar.timegm(timestamp)
        lines.append([timestamp] + parts[1:])

start_time = lines[0][0]
end_time = lines[-1][0]

# Make graph of # labels over time
i = 0
latest_query_type = None
knn_is_internal = False
last_seen_training = 0
last_incremented = 0

with open(OUTPUT_FILENAME, "w") as f:
    for t in range(math.floor(start_time), math.ceil(end_time) + 1):
        while i < len(lines) and lines[i][0] < t:
            # Parse this line
            l = lines[i]
            timestamp, activity_type = l[:2]
            if activity_type == "QUERY":
                latest_query_type = l[2]
            elif activity_type == "TRAINING STATUS":
                last_seen_training = timestamp
            elif activity_type == "INTERNAL KNN":
                knn_is_internal = True
            elif activity_type == "EXTERNAL BOOTSTRAPPING":
                knn_is_internal = False
            elif activity_type == "VALIDATION" and l[2] == "STACK":
                latest_query_type = "VALIDATION"
            i += 1
            last_incremented = t
        if t - last_incremented > PAUSE_THRESHOLD:
            continue
        query_type = latest_query_type
        if query_type == "KNN":
            query_type = "INTERNAL KNN" if knn_is_internal else "EXTERNAL BOOTSTRAPPING"
        is_training = t - last_seen_training < TRAINING_THRESHOLD
        f.write(f"{query_type}{'*' if is_training else ''}\n")
