from django.db import models


class Dataset(models.Model):
    label_function = models.CharField(max_length=300)
    dataset = models.CharField(max_length=300)
    data_idx = models.CharField(max_length=300)
    label_type = models.CharField(max_length=300)
    label = models.CharField(max_length=300)


class DatasetItem(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    identifier = models.CharField(max_length=300)
    label_function = models.CharField(max_length=300)
    dataset = models.CharField(max_length=300)
    label_type = models.CharField(max_length=300)
    label = models.CharField(max_length=300)


class Label(models.Model):
    label_function = models.CharField(max_length=300)
    dataset = models.CharField(max_length=300)
    data_idx = models.CharField(max_length=300)
    label_type = models.CharField(max_length=300)
    label = models.CharField(max_length=300)
