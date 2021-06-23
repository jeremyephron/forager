from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import Category, Mode, Annotation, CategoryCount

@receiver(post_save, sender=Annotation)
def increment_category_count(sender, **kwargs):
    category_count, _ = CategoryCount.objects.get_or_create(
        dataset=sender.dataset_item.dataset,
        category=sender.category,
        mode=sender.mode
    )
    category_count.count += 1
    category_count.save()


@receiver(post_delete, sender=Annotation)
def decrement_category_count(sender, **kwargs):
    category_count, _ = CategoryCount.objects.get_or_create(
        dataset=sender.dataset_item.dataset,
        category=sender.category,
        mode=sender.mode
    )
    category_count.count -= 1
    category_count.save()

# for dataset in Dataset.objects.filter():
#     x = Annotation.objects.filter(dataset_item__in=dataset.datasetitem_set.filter()).values("category__name", "mode__name").annotate(n=Count("pk"))
#     for c in x:
#         cat = Category.objects.get(name=c["category__name"])
#         mode = Mode.objects.get(name=c["mode__name"])
#         print(dataset.name, cat.name, mode.name, c["n"])
#         category_count, _ = CategoryCount.objects.get_or_create(
#             dataset=dataset,
#             category=cat,
#             mode=mode
#         )
#         category_count.count = c["n"]
#         category_count.save()
#     break
