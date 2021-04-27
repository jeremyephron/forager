from google.cloud import storage
from forager_server_api.models import Dataset, DatasetItem
import os

DATASET_NAME = "waymo"
VAL_IMAGE_DIR = "gs://foragerml/waymo/val"

if not VAL_IMAGE_DIR.startswith("gs://"):
    raise RuntimeError(
        "Directory only supports Google Storage bucket paths. "
        'Please specify as "gs://bucket-name/path/to/data".'
    )


# Similar to /create_dataset endpoint
split_dir = VAL_IMAGE_DIR[len("gs://") :].split("/")
bucket_name = split_dir[0]
bucket_path = "/".join(split_dir[1:])

client = storage.Client()
bucket = client.get_bucket(bucket_name)
all_blobs = client.list_blobs(bucket, prefix=bucket_path)

dataset = Dataset.objects.get(name=DATASET_NAME)
dataset.val_directory = VAL_IMAGE_DIR
dataset.save()

# Create all the DatasetItems for this dataset
paths = [blob.name for blob in all_blobs]
paths = [
    path
    for path in paths
    if (path.endswith(".jpg") or path.endswith(".jpeg") or path.endswith(".png"))
]

items = [
    DatasetItem(
        dataset=dataset,
        identifier=os.path.basename(path).split(".")[0],
        path=path,
        is_val=True,
    )
    for path in paths
]
print(f"Creating {len(items)} new entries")

DatasetItem.objects.bulk_create(items)
