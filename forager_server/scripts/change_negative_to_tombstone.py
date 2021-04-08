from forager_server_api.models import Dataset, Annotation
from enum import IntEnum
import json


DATASET_NAME = "waymo_train_central"


class LabelValue(IntEnum):
    TOMBSTONE = -1
    POSITIVE = 1
    NEGATIVE = 2
    HARD_NEGATIVE = 3
    UNSURE = 4


dataset = Dataset.objects.get(name=DATASET_NAME)
annotations = Annotation.objects.filter(
    dataset_item__in=dataset.datasetitem_set.filter(),
    label_type="klabel_frame",
)


for i, ann in enumerate(annotations):
    if i % 100 == 0:
        print(i)
    data = json.loads(ann.label_data)
    if data["value"] == LabelValue.NEGATIVE:
        data["value"] = LabelValue.TOMBSTONE
        ann.label_data = json.dumps(data)
        ann.save()
