from django.db import models


class Dataset(models.Model):
    name = models.SlugField(unique=True)
    directory = models.CharField(max_length=600)
    index_id = models.CharField(max_length=300, blank=True)


class DatasetItem(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    identifier = models.CharField(max_length=300)
    path = models.CharField(max_length=600)
    google = models.BooleanField(default=False)


class Annotation(models.Model):
    dataset_item = models.ForeignKey(DatasetItem, on_delete=models.CASCADE)
    label_function = models.CharField(max_length=300)
    label_category = models.CharField(max_length=300)
    label_type = models.CharField(max_length=300)
    label_data = models.CharField(max_length=5000)
    created = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)


class EmbeddingSet(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    index_name = models.CharField(max_length=1000)
