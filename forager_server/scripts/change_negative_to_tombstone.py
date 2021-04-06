from forager_server_api.models import Dataset, Annotation


def run():
    dataset = Dataset.objects.get(name="waymo_train_central")
    annotations = Annotation.objects.filter(
        dataset_item__in=dataset.datasetitem_set.filter(),
        label_type="klabel_frame",
    )
    print(len(annotations))
