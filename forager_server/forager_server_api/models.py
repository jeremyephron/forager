from django.db import models


MEDIUM_STRING_LENGTH = 300
LONG_STRING_LENGTH = 600


class Dataset(models.Model):
    name = models.SlugField(unique=True)
    train_directory = models.CharField(max_length=LONG_STRING_LENGTH)
    val_directory = models.CharField(max_length=LONG_STRING_LENGTH, blank=True)
    index_id = models.CharField(max_length=MEDIUM_STRING_LENGTH)
    hidden = models.BooleanField(default=False)

    class Meta:
        indexes = [
            models.Index(fields=["name"]),
        ]


class User(models.Model):
    email = models.EmailField(unique=True)

    class Meta:
        indexes = [
            models.Index(fields=["email"]),
        ]


class DatasetItem(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    identifier = models.CharField(max_length=MEDIUM_STRING_LENGTH)
    path = models.CharField(max_length=LONG_STRING_LENGTH)
    is_val = models.BooleanField(default=False)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "identifier"], name="unique_identifier"
            ),
        ]
        indexes = [
            models.Index(fields=["dataset", "is_val", "identifier"]),
        ]


class Category(models.Model):
    name = models.CharField(max_length=MEDIUM_STRING_LENGTH, unique=True)

    class Meta:
        indexes = [
            models.Index(fields=["name"]),
        ]


class Mode(models.Model):
    name = models.CharField(max_length=MEDIUM_STRING_LENGTH, unique=True)

    class Meta:
        indexes = [
            models.Index(fields=["name"]),
        ]


class Annotation(models.Model):
    dataset_item = models.ForeignKey(DatasetItem, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    mode = models.ForeignKey(Mode, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    bbox_x1 = models.FloatField(null=True)
    bbox_y1 = models.FloatField(null=True)
    bbox_x2 = models.FloatField(null=True)
    bbox_y2 = models.FloatField(null=True)
    misc_data = models.JSONField(default=dict)

    class Meta:
        indexes = [
            models.Index(fields=["dataset_item", "category", "mode"]),
        ]


class DNNModel(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    name = models.SlugField()
    model_id = models.CharField(max_length=MEDIUM_STRING_LENGTH, unique=True)
    checkpoint_path = models.CharField(max_length=LONG_STRING_LENGTH, null=True)
    output_directory = models.CharField(max_length=LONG_STRING_LENGTH, null=True)
    last_updated = models.DateTimeField(auto_now=True)
    category_spec = models.JSONField(default=dict)
    resume_model_id = models.CharField(max_length=MEDIUM_STRING_LENGTH, null=True)
    epoch = models.IntegerField(default=0)
