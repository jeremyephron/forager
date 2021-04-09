from forager_server_api.models import Dataset, Annotation
import json


DATASET_NAME = "waymo_train_central"
SRC_CATEGORY = ""
DST_CATEGORIES = ["", "", ""]

assert SRC_CATEGORY and all(DST_CATEGORIES)


dataset = Dataset.objects.get(name=DATASET_NAME)
annotations = (
    Annotation.objects.filter(
        dataset_item__in=dataset.datasetitem_set.filter(),
        label_category=SRC_CATEGORY,
        label_type="klabel_frame",
    )
    .order_by("dataset_item", "-created")
    .distinct("dataset_item")
)


new_anns = []
for DST_CATEGORY in DST_CATEGORIES:
    for ann in annotations:
        data = json.loads(ann.label_data)
        data["mode"] = "tag-duplicate"
        new_ann = Annotation(
            dataset_item=ann.dataset_item,
            label_function=ann.label_function,
            label_category=DST_CATEGORY,
            label_type=ann.label_type,
            label_data=json.dumps(data),
        )
        new_anns.append(new_ann)


Annotation.objects.bulk_create(new_anns)
