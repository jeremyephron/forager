from forager_server_api.models import Dataset, Annotation
from enum import IntEnum
import json


DATASET_NAME = "waymo_train_central"
OLD_TO_NEW_CATEGORIES = {
    "no zebra crossing": "zebra crossing",
    "not bay bridge": "bay bridge",
    "not taxi": "taxi",
}
ANN_VERSION = "2.0.1"


class LabelValue(IntEnum):
    TOMBSTONE = -1
    POSITIVE = 1
    NEGATIVE = 2
    HARD_NEGATIVE = 3
    UNSURE = 4


dataset = Dataset.objects.get(name=DATASET_NAME)
annotations = Annotation.objects.filter(
    dataset_item__in=dataset.datasetitem_set.filter(),
    label_category__in=list(OLD_TO_NEW_CATEGORIES.keys()),
    label_type="klabel_frame",
)

replacement_body = json.dumps(
    {
        "type": 0,  # full-frame
        "value": LabelValue.NEGATIVE,
        "mode": "tag-migrate",
        "version": ANN_VERSION,
    }
)

new_anns = []
for i, ann in enumerate(annotations):
    if i % 100 == 0:
        print(i)

    data = json.loads(ann.label_data)
    if data["value"] != LabelValue.POSITIVE:
        continue

    new_anns.append(
        Annotation(
            dataset_item=ann.dataset_item,
            label_function=ann.label_function,
            label_category=OLD_TO_NEW_CATEGORIES[ann.label_category],
            label_type="klabel_frame",
            label_data=replacement_body,
        )
    )

Annotation.objects.bulk_create(new_anns)
