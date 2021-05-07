from forager_server_api.models import Dataset, DatasetItem, Annotation
from django.utils import timezone

from collections import defaultdict
from datetime import datetime
from enum import IntEnum
import calendar
import time
import math
import json

from tqdm import tqdm

OUTPUT_FILENAME = "labels.txt"

LOG_FILENAME = "fait-taxi-failed.log"
CATEGORY = "fait taxi"
DATASET_NAME = "waymo"
PAUSE_THRESHOLD = 120  # seconds


class LabelValue(IntEnum):
    TOMBSTONE = -1
    POSITIVE = 1
    NEGATIVE = 2
    HARD_NEGATIVE = 3
    UNSURE = 4


def nest_anns(anns, nest_category=True, nest_lf=True):
    if nest_category and nest_lf:
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            c = ann.label_category
            u = ann.label_function
            data[k][c][u].append(ann)
    elif nest_category:
        data = defaultdict(lambda: defaultdict(list))
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            c = ann.label_category
            data[k][c].append(ann)
    elif nest_lf:
        data = defaultdict(lambda: defaultdict(list))
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            u = ann.label_function
            data[k][u].append(ann)
    else:
        data = defaultdict(list)
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            data[k].append(ann)
    return data


def filter_fn(anns):
    filt_anns = []
    most_recent = None
    for ann in anns:
        if ann.label_type == "klabel_frame":
            if most_recent is None or ann.created > most_recent.created:
                most_recent = ann
        else:
            filt_anns.append(ann)
    if most_recent:
        filt_anns.append(most_recent)
    return filt_anns


def filter_most_recent_anns(nested_anns):
    if len(nested_anns) == 0:
        return {}
    if isinstance(next(iter(nested_anns.items()))[1], list):
        data = defaultdict(list)
        for pk, anns in nested_anns.items():
            data[pk] = filter_fn(anns)
    elif isinstance(next(iter(next(iter(nested_anns.items()))[1].items()))[1], list):
        data = defaultdict(lambda: defaultdict(list))
        for pk, label_fns_data in nested_anns.items():
            for label_fn, anns in label_fns_data.items():
                data[pk][label_fn] = filter_fn(anns)
    elif isinstance(
        next(iter(next(iter(next(iter(nested_anns.items()))[1].items()))[1].items()))[
            1
        ],
        list,
    ):
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for pk, cat_fns_data in nested_anns.items():
            for cat, label_fns_data in cat_fns_data.items():
                for label_fn, anns in label_fns_data.items():
                    data[pk][cat][label_fn] = filter_fn(anns)
    return data


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
format_time = lambda t: datetime.fromtimestamp(t, tz=timezone.utc)

# Get all labels up to end time
dataset = Dataset.objects.get(name=DATASET_NAME)
eligible_images = DatasetItem.objects.filter(
    dataset=dataset, google=False, is_val=False
)
annotations = list(
    Annotation.objects.filter(
        dataset_item__in=eligible_images,
        label_category=CATEGORY,
        label_type="klabel_frame",
        created__lt=format_time(end_time),
    ).order_by("created")
)

# Make graph of # labels over time
i = 0
j = 0
last_incremented = 0

with open(OUTPUT_FILENAME, "w") as f:
    for t in tqdm(range(math.floor(start_time), math.ceil(end_time) + 1)):
        while j < len(lines) and lines[j][0] < t:
            j += 1
            last_incremented = t
        timeobj = format_time(t)
        while i < len(annotations) and annotations[i].created < timeobj:
            i += 1
        if t - last_incremented > PAUSE_THRESHOLD:
            continue
        num_neg = 0
        num_pos = 0
        anns = filter_most_recent_anns(nest_anns(annotations[:i], nest_lf=False))
        for di_pk, anns_by_cat in anns.items():
            for cat, ann_list in anns_by_cat.items():
                for ann in ann_list:
                    data = json.loads(ann.label_data)
                    if data["value"] == LabelValue.NEGATIVE:
                        num_neg += 1
                    elif data["value"] == LabelValue.POSITIVE:
                        num_pos += 1
        x = f.write(f"{num_pos} {num_neg}\n")
