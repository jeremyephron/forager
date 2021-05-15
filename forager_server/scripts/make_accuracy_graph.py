import numpy as np
import base64
from enum import IntEnum
import io
from collections import defaultdict, namedtuple
import itertools
from pathlib import Path
import json
from sklearn.metrics import f1_score
from forager_server_api.models import Dataset, DatasetItem, Annotation
import calendar
import time
import math

OUTPUT_FILENAME = "accuracy.txt"
PAUSE_THRESHOLD = 120  # seconds

DATASET_NAME = "waymo"
INDEX_DIR = Path("/home/fpoms/forager/indexes/2d2b13f9-3b30-4e51-8ab9-4e8a03ba1f03")
MODEL_OUTPUTS_PARENT_DIR = Path("~/forager/model_outputs").expanduser().resolve()
EMBEDDING_DIM = 2048
CATEGORY = "zebra crossing"
LOG_FILENAME = "zebra1.log"

VAL_IDENTIFIERS = json.load((INDEX_DIR / "val_identifiers.json").open())
ALL_LABELS = json.load((INDEX_DIR / "labels.json").open())


class LabelValue(IntEnum):
    TOMBSTONE = -1
    POSITIVE = 1
    NEGATIVE = 2
    HARD_NEGATIVE = 3
    UNSURE = 4
    CUSTOM = 5


def base64_to_numpy(nda_base64):
    if not nda_base64:
        return None
    nda_bytes = base64.b64decode(nda_base64)
    with io.BytesIO(nda_bytes) as nda_buffer:
        nda = np.load(nda_buffer, allow_pickle=False)
    return nda


Tag = namedtuple("Tag", "category value")


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


def parse_tag_set_from_query_v2(s):
    if isinstance(s, list):
        parts = s
    elif isinstance(s, str) and s:
        parts = s.split(",")
    else:
        parts = []
    ts = set()
    for part in parts:
        if not part:
            continue
        category, value_str = part.split(":")
        ts.add(Tag(category, value_str))
    return ts


def tag_sets_to_category_list_v2(*tagsets):
    categories = set()
    for ts in tagsets:
        for tag in ts:
            categories.add(tag.category)
    return list(categories)


def serialize_tag_set_for_client_v2(ts):
    return [{"category": t.category, "value": t.value} for t in sorted(list(ts))]


def get_tags_from_annotations_v2(annotations):
    # [image][category][#]
    anns = filter_most_recent_anns(nest_anns(annotations, nest_lf=False))
    tags_by_pk = defaultdict(list)
    for di_pk, anns_by_cat in anns.items():
        for cat, ann_list in anns_by_cat.items():
            if not cat:
                continue
            assert len(ann_list) == 1  # should only be one latest per-frame annotation
            ann = ann_list[0]
            label_data = json.loads(ann.label_data)
            value = LabelValue(label_data["value"])
            if value == LabelValue.TOMBSTONE:
                continue
            value_str = (
                label_data["custom_value"] if value == LabelValue.CUSTOM else value.name
            )
            tags_by_pk[di_pk].append(Tag(cat, value_str))
    return tags_by_pk


def get_val_examples_v2(dataset):
    # Get positive and negative categories
    pos_tags = set([Tag(CATEGORY, LabelValue.POSITIVE.name)])
    neg_tags = set([Tag(CATEGORY, LabelValue.NEGATIVE.name)])
    # Limit to validation set
    eligible_dataset_items = DatasetItem.objects.filter(
        dataset=dataset,
        google=False,
        is_val=True,
    )
    # Get positives and negatives matching these categories
    annotations = Annotation.objects.filter(
        dataset_item__in=eligible_dataset_items,
        label_category__in=tag_sets_to_category_list_v2(pos_tags, neg_tags),
        label_type="klabel_frame",
    )
    tags_by_pk = get_tags_from_annotations_v2(annotations)
    pos_dataset_item_pks = []
    neg_dataset_item_pks = []
    for pk, tags in tags_by_pk.items():
        if any(t in pos_tags for t in tags):
            pos_dataset_item_pks.append(pk)
        elif any(t in neg_tags for t in tags):
            neg_dataset_item_pks.append(pk)
    return pos_dataset_item_pks, neg_dataset_item_pks


# Get validation labels
dataset = Dataset.objects.get(name=DATASET_NAME)
pos_dataset_item_pks, neg_dataset_item_pks = get_val_examples_v2(dataset)

# Construct paths, identifiers, and labels
dataset_items_by_pk = DatasetItem.objects.in_bulk(
    pos_dataset_item_pks + neg_dataset_item_pks
)
identifiers = []
labels = []
for pk, label in itertools.chain(
    ((pk, True) for pk in pos_dataset_item_pks),
    ((pk, False) for pk in neg_dataset_item_pks),
):
    di = dataset_items_by_pk[pk]
    identifiers.append(di.identifier)
    labels.append(label)

# Read log file
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


def get_svm_accuracy(vector_b64, model, val_identifiers, val_labels):
    vector = base64_to_numpy(vector_b64)
    # Read embeddings from disk
    embeddings = np.memmap(
        str(INDEX_DIR / "local" / model / "embeddings.npy"),
        dtype="float32",
        mode="r",
        shape=(len(ALL_LABELS), EMBEDDING_DIM),
    )
    # Get relevant embeddings
    inds = []
    for identifier in val_identifiers:
        inds.append(VAL_IDENTIFIERS[identifier])
    relevant_embeddings = embeddings[inds]
    # Compute scores
    scores = relevant_embeddings @ vector
    # Compute labels
    print(sum(scores > 0))
    return f1_score(val_labels, scores > 0)


def get_dnn_accuracy(model, val_identifiers, val_labels):
    all_scores = np.load(str(MODEL_OUTPUTS_PARENT_DIR / model / "scores.npy"))
    # Get relevant scores
    inds = []
    for identifier in val_identifiers:
        inds.append(VAL_IDENTIFIERS[identifier])
    scores = all_scores[inds]
    # Compute labels
    return f1_score(val_labels, scores > 0.5)


# Make graph of # labels over time
i = 0
best_accuracy = 0
seen_dnns = set()
last_incremented = 0

with open(OUTPUT_FILENAME, "w") as f:
    for t in range(math.floor(start_time), math.ceil(end_time) + 1):
        while i < len(lines) and lines[i][0] < t:
            # Parse this line
            l = lines[i]
            timestamp, activity_type = l[:2]
            if activity_type == "NEW SVM":
                svm_accuracy = get_svm_accuracy(
                    l[2], l[3] if len(l) > 3 else "imagenet", identifiers, labels
                )
                best_accuracy = max(best_accuracy, svm_accuracy)
            elif activity_type == "NEW AVAILABLE DNN":
                dnn_id = l[2]
                if dnn_id not in seen_dnns:
                    dnn_accuracy = get_dnn_accuracy(dnn_id, identifiers, labels)
                    best_accuracy = max(best_accuracy, dnn_accuracy)
                    seen_dnns.add(dnn_id)
            i += 1
            last_incremented = t
        if t - last_incremented > PAUSE_THRESHOLD:
            continue
        f.write(f"{best_accuracy}\n")
