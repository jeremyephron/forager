from forager_server_api.models import Dataset

OLD_NAME = "waymo_train_central"
NEW_NAME = "waymo"

dataset = Dataset.objects.get(name=OLD_NAME)
dataset.name = NEW_NAME
dataset.save()
