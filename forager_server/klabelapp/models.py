from django.db import models


class Dataset(models.Model):
    name = models.SlugField()
    directory = models.CharField(max_length=600)


class DatasetItem(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    identifier = models.CharField(max_length=300)
    path = models.CharField(max_length=600)


class Label(models.Model):
    dataset_item = models.ForeignKey(DatasetItem, on_delete=models.CASCADE)
    label_function = models.CharField(max_length=300)
    label_type = models.CharField(max_length=300)
    label = models.CharField(max_length=300)
