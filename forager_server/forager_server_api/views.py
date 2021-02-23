import distutils.util
from google.cloud import storage
from enum import Enum
import itertools
import json
import os
import requests
import urllib.request

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.db.models import Q
from django.core.exceptions import ValidationError, ObjectDoesNotExist
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_object_or_404
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from collections import defaultdict
from pycocotools.coco import COCO

from .models import Dataset, DatasetItem, Annotation

class LabelType(Enum):
    positive = 1
    negative = 2
    hard_negative = 3
    unsure = 4

def nest_anns(anns, nest_category=True, nest_lf=True):
    if nest_category and nest_lf:
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            c = ann.label_category
            u = ann.label_function
            data[k][c][u].append(ann)
    elif nest_category:
        data = defaultdict(lambda: defaultdict(list))
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            c = ann.label_category
            data[k][c].append(ann)
    elif nest_lf:
        data = defaultdict(lambda: defaultdict(list))
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            u = ann.label_function
            data[k][u].append(ann)
    else:
        data = defaultdict(list)
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            data[k].append(ann)
    return data


def filter_most_recent_anns(nested_anns):
    def filter_fn(anns):
        filt_anns = []
        most_recent = None
        for ann in anns:
            if ann.label_type == 'klabel_frame':
                if most_recent is None or ann.created > most_recent.created:
                    most_recent = ann
                else:
                    filt_anns.append(ann)
        if most_recent:
            filt_anns.append(most_recent)
        return filt_anns

    if len(nested_anns) == 0:
        return {}
    if isinstance(next(iter(nested_anns.items()))[1], list):
       data = defaultdict(list)
       for pk, anns in nested_anns.items():
           data[pk] = filter_fn(anns)
    elif isinstance(
            next(iter(
                next(iter(nested_anns.items()))[1].items()
            ))[1],
            list):
       data = defaultdict(lambda: defaultdict(list))
       for pk, label_fns_data in nested_anns.items():
           for label_fn, anns in label_fns_data.items():
               data[pk][label_fn] = filter_fn(anns)
    elif isinstance(
            next(iter(
                next(iter(
                    next(iter(nested_anns.items()))[1].items()
                ))[1].items()
            ))[1],
            list):
       data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
       for pk, cat_fns_data in nested_anns.items():
           for cat, label_fns_data in cat_fns_data.items():
               for label_fn, anns in label_fns_data.items():
                   data[pk][cat][label_fn] = filter_fn(anns)
    return data


def aggregate_frame_anns_majority(anns):
    agg_anns = {}
    for img_id, data in anns.items():
        label_values = defaultdict(int)
        for label_function, ann in data.items():
            label_values[json.loads(ann.label_data)['value']] += 1
        max_k = None
        max_c = -1
        for k, c in label_values.items():
            if c > max_c:
                max_k = k
                max_c = c
        agg_anns[img_id] = max_k
    return agg_anns


@api_view(['GET'])
@csrf_exempt
def get_datasets(request):
    datasets = Dataset.objects.filter()

    def serialize(dataset):
        # JEB: probably should make a real serializer
        n_labels = 0
        if getattr(dataset.datasetitem_set, 'annotation_set', None):
            n_labels = dataset.datasetitem_set.annotation_set.count()

        return {
            'name': dataset.name,
            'size': dataset.datasetitem_set.count(),
            'n_labels': n_labels,
            'last_labeled': 'N/A'
        }

    return JsonResponse(list(map(serialize, datasets)), safe=False)


@api_view(['POST'])
@csrf_exempt
def start_cluster(request):
    # TODO(mihirg): Remove this setting from Django; it's now managed by Terraform
    # (or figure out how to set it from the frontend if we need that)
    params = {"n_nodes": settings.EMBEDDING_CLUSTER_NODES}
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/start_cluster",
        json=params,
    )
    response_data = r.json()
    return JsonResponse({
        "status": "success",
        "cluster_id": response_data["cluster_id"],
    })


@api_view(['GET'])
@csrf_exempt
def get_cluster_status(request, cluster_id):
    params = {"cluster_id": cluster_id}
    r = requests.get(
        settings.EMBEDDING_SERVER_ADDRESS + "/cluster_status", params=params
    )
    response_data = r.json()
    return JsonResponse(response_data)


@api_view(['POST'])
@csrf_exempt
def stop_cluster(request, cluster_id):
    params = {"cluster_id": cluster_id}
    requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/stop_cluster",
        json=params,
    )
    return JsonResponse({
        "status": "success",
    })


@api_view(['POST'])
@csrf_exempt
def create_index(request, dataset_name, dataset=None):
    if not dataset:
        dataset = get_object_or_404(Dataset, name=dataset_name)

    bucket_name = dataset.directory[len('gs://'):].split('/')[0]
    dataset_items = DatasetItem.objects.filter(dataset=dataset,google=False)
    dataset_item_raw_paths = [di.path for di in dataset_items]
    dataset_item_identifiers = [di.identifier for di in dataset_items]

    data = json.loads(request.body)
    params = {
        "cluster_id": data["cluster_id"],
        "bucket": bucket_name,
        "paths": dataset_item_raw_paths,
        "identifiers": dataset_item_identifiers,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/start_job",
        json=params,
    )
    response_data = r.json()
    return JsonResponse({
        "status": "success",
        "index_id": response_data["index_id"],
    })


@api_view(['GET'])
@csrf_exempt
def get_index_status(request, index_id):
    dataset_name = request.GET['dataset']

    params = {"index_id": index_id}
    r = requests.get(
        settings.EMBEDDING_SERVER_ADDRESS + "/job_status", params=params
    )
    response_data = r.json()

    # Comment this out to prevent uploading and storage
    if response_data["has_index"]:
        # Index has been successfully created & uploaded -> persist
        dataset = get_object_or_404(Dataset, name=dataset_name)
        dataset.index_id = index_id
        dataset.save()

    return JsonResponse(response_data)


@api_view(['POST'])
@csrf_exempt
def download_index(request, index_id):
    params = {"index_id": index_id}
    requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/download_index",
        json=params,
    )
    return JsonResponse({
        "status": "success",
    })


@api_view(['POST'])
@csrf_exempt
def delete_index(request, index_id):
    params = {"index_id": index_id}
    requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/delete_index",
        json=params,
    )
    return JsonResponse({
        "status": "success",
    })


@api_view(['POST'])
@csrf_exempt
def create_dataset(request):
    try:
        data = json.loads(request.body)
        name = data['dataset']
        data_directory = data['dirpath']
        if not data_directory.startswith('gs://'):
            err = ValidationError(
                'Directory only supports Google Storage bucket paths. '
                'Please specify as "gs://bucket-name/path/to/data".')
            raise err

        split_dir = data_directory[len('gs://'):].split('/')
        bucket_name = split_dir[0]
        bucket_path = '/'.join(split_dir[1:])

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        all_blobs = client.list_blobs(bucket, prefix=bucket_path)

        dataset = Dataset(name=name, directory=data_directory)
        dataset.save()

        # Create all the DatasetItems for this dataset
        paths = [blob.name for blob in all_blobs]
        paths = [path for path in paths
                 if (path.endswith('.jpg') or
                     path.endswith('.jpeg') or
                     path.endswith('.png'))]

        # these paths do not have a prefix--probably can just load images differently if with prefix
        items = [
            DatasetItem(dataset=dataset,
                        identifier=os.path.basename(path).split('.')[0],
                        path=path)
            for path in paths
        ]
        DatasetItem.objects.bulk_create(items, batch_size=10000)

        req = HttpRequest()
        req.method = 'GET'
        return get_dataset_info(req, name, dataset)
    except ValidationError:
        return JsonResponse({
            'status': 'failure',
            'message': 'Something went wrong. Make sure the directory path is valid.',
        })

@api_view(['POST'])
@csrf_exempt
def import_annotations(request, dataset_name, dataset = None):
    try:
        if not dataset:
            dataset = get_object_or_404(Dataset, name=dataset_name)

        print(request)
        data = json.loads(request.body)

        # Download annotations to local tmp directory--should add a way to check if the file is already present?
        ann_file = data['ann_file']

        file_name, headers = urllib.request.urlretrieve(ann_file)

        print("Annotations stored at: " + file_name)

        coco=COCO(file_name) # Only supporting coco-style annotation files for now (this includes inaturalist!)

        # get all images containing given categories, select one at random
        category = data['category']
        catIds = coco.getCatIds(catNms=[category])
        imgIds = coco.getImgIds(catIds=catIds)
        annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds)
        anns = coco.loadAnns(annIds)

        strImgIds = [os.path.basename(coco.loadImgs(imgId)[0]['file_name']).split('.')[0] for imgId in imgIds] # np.array(imgIds).astype('str').tolist()
        print("Have image ids for category: " + strImgIds[0])
        print(strImgIds)
        dataset_items = DatasetItem.objects.filter(dataset=dataset,google=False,identifier__in=strImgIds)
        print(len(dataset_items))
        # Create a map from identifier to pk
        item_identifiers = [di.identifier for di in dataset_items]
        item_pks = [di.pk for di in dataset_items]
        item_dict = dict(zip(item_identifiers, item_pks))

        returnAnns = []
        print(anns)
        # Check if bbox or just categories
        if ('bbox' in anns[0]):
            # Add bounding box labels, assume per-frame labels already loaded
            for ann in anns:
                img = coco.loadImgs(ann['image_id'])[0]
                imgWidth = img['width'] #x
                imgHeight = img['height'] #y
                bbox = ann['bbox']
                print(bbox)
                xmin = bbox[0]/imgWidth
                ymin = bbox[1]/imgHeight
                xmax = (bbox[0] + bbox[2])/imgWidth
                ymax = (bbox[1] + bbox[3])/imgHeight
                print(xmin, ymin, xmax, ymax)
                label_function = 'ground_truth' # Name associated with ground-truth annotations
                label_type = 'klabel_box'
                # Type: 2 means box, need to compute bbox on 0-1 scale
                annotation = {
                    'type': 2,
                    'bbox': {
                        'bmin': {
                            'x':xmin,
                            'y':ymin
                        },
                        'bmax': {
                            'x':xmax,
                            'y':ymax
                        }
                    }
                }
                annotation = json.dumps(annotation)
                identifier = os.path.basename(img['file_name']).split('.')[0]
                dataset_item = DatasetItem.objects.get(pk=item_dict[identifier])
                ann = Annotation(
                    dataset_item=dataset_item,
                    label_function=label_function,
                    label_category=category,
                    label_type=label_type,
                    label_data=annotation)
                returnAnns.append(ann)
        else:
            for dataset_item in dataset_items:
                label_function = 'ground_truth' # Name associated with ground-truth annotations
                label_type = 'klabel_frame'
                # Type: 0 means full-frame, value: 1 means positive
                annotation = {
                    'type': 0,
                    'value': 1
                }
                annotation = json.dumps(annotation)
                ann = Annotation(
                    dataset_item=dataset_item,
                    label_function=label_function,
                    label_category=category,
                    label_type=label_type,
                    label_data=annotation)
                returnAnns.append(ann)

        print(len(returnAnns))
        Annotation.objects.bulk_create(returnAnns, batch_size=10000)
        print("Successful bulk create")

        return JsonResponse({
            'status': 'success'
        })
    except ValidationError:
        return JsonResponse({
            'status': 'failure',
            'message': 'Something went wrong. Make sure the directory path is valid.',
        })

def query_google(request, dataset):
    category = request.GET['category']
    start = int(request.GET.get('start', 1))

    # Try getting data from google here
    key = os.environ["SEARCH_KEY"]
    engine_id = os.environ["SEARCH_ENGINE"]

    params = {"key": key, "cx": engine_id, "q": category, "start": start, "searchType": "image"}
    r = requests.get(
        "https://www.googleapis.com/customsearch/v1", params=params
    )
    response_data = r.json()
    paths = [item["link"] for item in response_data["items"]]
    num_total = int(response_data["searchInformation"]["totalResults"])

    # Check which paths are already in the dataset
    existing_items = DatasetItem.objects.filter(dataset=dataset,path__in=paths).order_by('pk')
    existing_paths = [di.path for di in existing_items]

    # How to generate identifiers
    # Add items, then get auto-generated identifiers
    identifiers = []
    for path in paths:
        if (path not in existing_paths):
            newItem = DatasetItem.objects.create(dataset=dataset,identifier="",path=path,google=True)
            identifiers.append(newItem.pk)
        else:
            pathIndex = existing_paths.index(path)
            identifiers.append(existing_items[pathIndex].pk)

    return [identifiers, paths, num_total]

@api_view(['GET'])
@csrf_exempt
def get_google(request, dataset_name, dataset = None):
    if not dataset:
        dataset = get_object_or_404(Dataset, name=dataset_name)
    try:
        identifiers, paths, num_total = query_google(request, dataset)

        return JsonResponse({
            'status': 'success',
            'identifiers': identifiers,
            'paths': paths,
            'num_total': num_total
        })
    except ValidationError:
        return JsonResponse({
            'status': 'failure',
            'message': 'Something went wrong. Make sure the directory path is valid.',
        })

@api_view(['GET'])
@csrf_exempt
def get_dataset_info(request, dataset_name, dataset=None):
    if not dataset:
        dataset = get_object_or_404(Dataset, name=dataset_name)

    bucket_name = dataset.directory[len('gs://'):].split('/')[0]
    path_template = 'https://storage.googleapis.com/{:s}/'.format(bucket_name) + '{:s}'
    dataset_items = DatasetItem.objects.filter(dataset=dataset).order_by('pk')[:500]
    dataset_item_paths = [path_template.format(di.path) for di in dataset_items]
    dataset_item_identifiers = [di.pk for di in dataset_items]

    return JsonResponse({
        'status': 'success',
        'datasetName': dataset.name,
        'indexId': dataset.index_id,
        'paths': dataset_item_paths,
        'identifiers': dataset_item_identifiers
    })


@api_view(['GET'])
@csrf_exempt
def get_users_and_categories(request, dataset_name):
    dataset = Dataset.objects.get(name=dataset_name)

    annotations = Annotation.objects.filter(
        dataset_item__in=dataset.datasetitem_set.filter())

    users = set()
    categories = set()
    for ann in annotations:
        users.add(ann.label_function)
        categories.add(ann.label_category)

    users = sorted(list(users))
    categories = sorted(list(categories))
    result = dict(
        users=users,
        categories=categories
    )

    return JsonResponse(result)

def filtered_images(request, dataset, path_filter=None):
    label_function = request.GET.get('user', '')
    category = request.GET.get('category', '')
    label_value = request.GET['filter']

    dataset_items = DatasetItem.objects.filter(dataset=dataset,google=False).order_by('pk')
    print(len(dataset_items))
    if path_filter:
        print("Filtering items")
        dataset_items = dataset_items.filter(path__in=path_filter)
        print(len(dataset_items))

    print(label_value)

    # Find all annotations for the user and category in this dataset
    annotations = Annotation.objects.filter(
        dataset_item__in=dataset_items,
        label_function=label_function,
        label_category=category,
        label_type='klabel_frame')

    next_images = []
    if label_value == 'all':
        next_images = dataset_items
    elif label_value == 'google':
        next_images = DatasetItem.objects.filter(dataset=dataset,google=True).order_by('pk')
    elif label_value == 'unlabeled':
        images_with_label = set()
        for ann in annotations:
            images_with_label.add(ann.dataset_item)

        # Get all images which are not in images_with_label
        for ditem in dataset_items:
            if ditem not in images_with_label:
                next_images.append(ditem)
    elif label_value == 'conflict':
        conflict_data = get_annotation_conflicts_helper(
            dataset_items, label_function, category)
        for ditem in dataset_items:
            if ditem.pk in conflict_data:
                next_images.append(ditem)
    else:
        anns = filter_most_recent_anns(
            nest_anns(annotations, nest_category=False, nest_lf=False))

        cat_value = LabelType[label_value].value
        images_with_label = set()
        for di_pk, ann_list in anns.items():
            for ann in ann_list:
                label_value = json.loads(ann.label_data)
                if label_value['value'] != cat_value:
                    continue
                images_with_label.add(ann.dataset_item)
        for ditem in dataset_items:
            if ditem in images_with_label:
                next_images.append(ditem)

    print("Post-filter:")
    print(len(next_images))

    return next_images

@api_view(['GET'])
@csrf_exempt
def get_next_images(request, dataset_name, dataset=None):
    if not dataset:
        dataset = get_object_or_404(Dataset, name=dataset_name)

    bucket_name = dataset.directory[len('gs://'):].split('/')[0]
    path_template = 'https://storage.googleapis.com/{:s}/'.format(bucket_name) + '{:s}'
    offset_to_return = int(request.GET.get('offset', 0))
    num_to_return = int(request.GET.get('num', 100))

    next_images = filtered_images(request, dataset)
    ret_images = next_images[offset_to_return:offset_to_return+num_to_return]

    dataset_item_paths = [(di.path if di.path.find("http") != -1 else path_template.format(di.path))
                          for di in ret_images]
    dataset_item_identifiers = [di.pk for di in ret_images]

    return JsonResponse({
        'paths': dataset_item_paths,
        'identifiers': dataset_item_identifiers,
        'num_total': len(next_images),
    })

@api_view(['GET'])
@csrf_exempt
def get_annotations(request, dataset_name):
    if len(request.GET['identifiers']) == 0:
        return JsonResponse({})

    image_identifiers = request.GET['identifiers'].split(',')
    label_function = request.GET['user']
    category = request.GET['category']

    dataset_items = DatasetItem.objects.filter(pk__in=image_identifiers)
    filter_args = dict(
        dataset_item__in=dataset_items
    )
    if not label_function == 'all':
        filter_args['label_function'] = label_function
    if not category == 'all':
        filter_args['label_category'] = category


    anns = Annotation.objects.filter(**filter_args)

    data = defaultdict(list)
    for ann in anns:
        label_data = json.loads(ann.label_data)
        label_data['identifier'] = ann.pk
        data[ann.dataset_item.pk].append(label_data)

    return JsonResponse(data)


@api_view(['GET'])
@csrf_exempt
def get_annotations_summary(request, dataset_name):
    dataset = Dataset.objects.get(name=dataset_name)
    total_images = dataset.datasetitem_set.filter(google=False).count()
    anns = Annotation.objects.filter(
        label_type="klabel_frame",
        dataset_item__in=dataset.datasetitem_set.filter(google=False))
    label_functions = [d['label_function'] for d in (anns.values("label_function").distinct())]
    label_categories = [d['label_category'] for d in (anns.values("label_category").distinct())]

    # For each (label_function, label_category) pair, get the number of
    # annotations

    # label_category -> label_function -> label_value -> total
    ann_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # label_function -> list of label_categories
    label_function_categories = defaultdict(set)
    # label_function -> label_value -> total
    label_function_values = defaultdict(lambda: defaultdict(int))
    # label_category -> label_value -> total
    label_category_values = defaultdict(lambda: defaultdict(int))
    if False:
        for label_function in label_functions:
            for label_category in label_categories:
                func_cat_anns = anns.filter(
                    label_function=label_function,
                    label_category=label_category)
                filtered_frame_anns = filter_most_recent_anns(
                    nest_anns(func_cat_anns))

                count = 0
                label_value_totals = defaultdict(int)
                # filter to most recent ann per image
                for ann in filtered_anns.values():
                    ann = ann[label_function]
                    # Add to ann_summary
                    label_value = json.loads(ann.label_data)['value']
                    ann_summary[label_category][label_function][label_value] += 1
                    label_value_totals[label_value] += 1
                    count += 1
                if count > 0:
                    label_function_categories[label_function].add(label_category)

                    # Add unlabeled as total images - count
                    label_value = 'unlabeled'
                    unlabeled_count = total_images - count
                    ann_summary[label_category][label_function][label_value] = \
                        unlabeled_count
                    label_value_totals[label_value] = unlabeled_count

                for label_value, total in label_value_totals.items():
                    label_function_values[label_function][label_value] += total
                    label_category_values[label_category][label_value] += total
    elif False:
        # For each user, find num unique annotations
        filtered_anns = filter_most_recent_anns(nest_anns(func_cat_anns))
        counts = defaultdict(lambda: defaultdict(int))
        for im_id, data in filtered_anns.items():
            for label_function, anns in data.items():
                for ann in anns:
                    label_category = ann.label_category
                    counts[label_category][label_function] += 1
                    label_value = json.loads(label_value)['value']
                    label_function_values[label_function][label_value] += 1
    else:
        for label_function in label_functions:
            for label_category in label_categories:
                num_unique_image_anns = (
                    anns.filter(
                        label_function=label_function,
                        label_category=label_category)
                    .values("dataset_item").distinct().count())
                num_unlabeled = total_images - num_unique_image_anns
                if num_unique_image_anns > 0:
                    ann_summary[label_category][label_function]['unlabeled'] += num_unlabeled
                    label_function_categories[label_function].add(label_category)
                    label_function_values[label_function]['unlabeled'] += num_unlabeled
                    label_category_values[label_category]['unlabeled'] += num_unlabeled


    data = {
        'data': ann_summary,
        'user_categories': {k: list(v)
                            for k, v in label_function_categories.items()},
        'user_totals': label_function_values,
        'category_totals': label_category_values,
    }


    return JsonResponse(data)


@api_view(['GET'])
@csrf_exempt
def dump_annotations(request, dataset_name):
    dataset = Dataset.objects.get(name=dataset_name)
    total_images = dataset.datasetitem_set.filter(google=False).count()
    anns = Annotation.objects.filter(
        dataset_item__in=dataset.datasetitem_set.filter(google=False))
    filtered_frame_anns = filter_most_recent_anns(
        nest_anns([ann for ann in anns
                   if ann.label_type == 'klabel_frame']))

    pk_to_img_id = {}
    for ann in anns:
        pk = ann.dataset_item.pk
        img_id = ann.dataset_item.identifier
        pk_to_img_id[pk] = img_id

    label_map = {
        1: 'pos',
        2: 'neg',
        3: 'hard_neg',
        4: 'unsure',
    }
    output_data = defaultdict(lambda: defaultdict(dict))
    for pk, cats_data in filtered_frame_anns.items():
        img_id = pk_to_img_id[pk]
        for cat, label_functions_data in cats_data.items():
            for label_function, anns in label_functions_data.items():
                label_value = None
                bboxes = []
                for ann in anns:
                    if ann.label_type == 'klabel_frame':
                        label_value = json.loads(ann.label_data)['value']
                    else:
                        label_data = json.loads(ann.label_data)
                        bbox_data = label_data['bbox']
                        bboxes.append(
                            (bbox_data['bmin']['x'], bbox_data['bmin']['y'],
                             bbox_data['bmax']['x'], bbox_data['bmax']['y']))
                output_data[img_id][cat][label_function] = {
                    'label': label_map[label_value],
                    'bboxes': bboxes,
                }
    # Output
    # { image_id:
    #   {label_function:
    #     {label: 'pos' OR 'neg' OR 'unsure',
    #      bboxes: [(top_left_x, top_left_y, bottom_right_x, bottom_right_y)]
    #     }
    #   }
    # }

    return JsonResponse(output_data)

def get_annotation_conflicts_helper(dataset_items, label_function, category):
    filter_args = dict(
        dataset_item__in=dataset_items,
        label_type='klabel_frame'
    )
    if not category == 'all':
        filter_args['label_category'] = category

    anns = Annotation.objects.filter(**filter_args)

    data = filter_most_recent_anns(nest_anns(anns, nest_category=False))

    # Analyze conflicts
    conflict_data = defaultdict(set)
    for image_id, user_labels in data.items():
        if label_function not in user_labels:
            continue
        user_annotations = user_labels[label_function]
        assert len(user_annotations) == 1
        user_annotation = user_annotations[0]
        user_label_value = json.loads(user_annotation.label_data)
        for label_function, anns in user_labels.items():
            if label_function == user_annotation.label_function:
                continue
            for ann in anns:
                label_value = json.loads(ann.label_data)
                if label_value['value'] != user_label_value['value']:
                    conflict_data[image_id].add(ann.label_function)
    return conflict_data


@api_view(['GET'])
@csrf_exempt
def get_annotation_conflicts(request, dataset_name):
    if len(request.GET['identifiers']) == 0:
        return JsonResponse({})

    image_identifiers = request.GET['identifiers'].split(',')
    label_function = request.GET['user']
    category = request.GET['category']

    dataset_items = DatasetItem.objects.filter(pk__in=image_identifiers)

    conflict_data = get_annotation_conflicts_helper(
        dataset_items, label_function, category)

    return JsonResponse({k: list(v) for k,v in conflict_data.items()})


@api_view(['POST'])
@csrf_exempt
def add_annotation(request, dataset_name, image_identifier):
    # body is the JSON of a single annotation
    # {
    #   'type': 2,
    #   'bbox': {'bmin': {'x': 0.2, 'y': 0.5}, 'bmax': {'x': 0.5, 'y': 0.7}}
    # }
    body = json.loads(request.body) # request.body.decode('utf-8')
    label_function = body['user']
    category = body['category']
    label_type = body['label_type']

    annotation = json.dumps(body['annotation'])

    dataset_item = DatasetItem.objects.get(pk=image_identifier)
    ann = Annotation(
        dataset_item=dataset_item,
        label_function=label_function,
        label_category=category,
        label_type=label_type,
        label_data=annotation)
    ann.save()

    ann_identifier = ann.pk
    return HttpResponse(ann_identifier)


@api_view(['DELETE'])
@csrf_exempt
def delete_annotation(request, dataset_name, image_identifier, ann_identifier):
    try:
        Annotation.objects.get(pk=ann_identifier).delete()
        return HttpResponse(status=status.HTTP_204_NO_CONTENT)
    except ObjectDoesNotExist as e:
        return JsonResponse(
            {'error': str(e)},
            safe=False,
            status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return JsonResponse(
            {'error': str(e)},
            safe=False,
            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def process_image_query_results(
    request,
    dataset,
    query_response,
    filter_func=filtered_images,
):
    ordered_results = query_response['results']

    data_directory = dataset.directory
    split_dir = data_directory[len('gs://'):].split('/')
    bucket_name = split_dir[0]

    ditems = filtered_images(request, dataset, [r['label'] for r in ordered_results])

    response = {
        'identifiers': [],
        'paths': [],
        'all_spatial_dists': [],
    }

    path_to_id = {}
    for ditem in ditems:
        path_to_id[ditem.path] = ditem.pk
    path_template = 'https://storage.googleapis.com/{:s}/'.format(bucket_name) + '{:s}'
    for result in ordered_results:
        path = result['label']
        if path in path_to_id:
            response['paths'].append(path_template.format(path))
            response['identifiers'].append(path_to_id[path])
            response['all_spatial_dists'].append(result['spatial_dists'])

    response['num_total'] = len(response['paths'])
    del query_response['results']
    response.update(query_response)  # include any other keys from upstream response

    return response

@api_view(['GET'])
@csrf_exempt
def lookup_knn(request, dataset_name):
    ann_identifiers = [int(x) for x in request.GET['ann_identifiers'].split(',')]
    cluster_id = request.GET['cluster_id']
    index_id = request.GET['index_id']
    augmentations = [x.split(":") for x in request.GET['augmentations'].split(',')]
    use_full_image = bool(distutils.util.strtobool(request.GET['use_full_image']))

    # 1. Retrieve dataset info from db
    dataset = Dataset.objects.get(name=dataset_name)
    data_directory = dataset.directory
    split_dir = data_directory[len('gs://'):].split('/')
    bucket_name = split_dir[0]
    bucket_path = '/'.join(split_dir[1:])

    # 2. Retrieve annotations from db
    query_annotations = Annotation.objects.filter(pk__in=ann_identifiers)
    paths = [ann.dataset_item.path for ann in query_annotations]
    window = float(request.GET['window'])
    patches = [{'x1': max(bbox['bmin']['x'] - window, 0.),
                'y1': max(bbox['bmin']['y'] - window, 0.),
                'x2': min(bbox['bmax']['x'] + window, 1.),
                'y2': min(bbox['bmax']['y'] + window, 1.)}
               for bbox in [json.loads(ann.label_data)['bbox']
                            for ann in query_annotations]]

    # 3. Send paths and patches to /query_index
    params = {
        "cluster_id": cluster_id,
        "index_id": index_id,
        "bucket": bucket_name,
        "paths": paths,
        "patches": patches,
        "augmentations": augmentations,
        "num_results": 100,
        "use_full_image": use_full_image,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_index",
        json=params,
    )
    response_data = r.json()
    response = process_image_query_results(request, dataset, response_data)

    # 4. Return knn results
    return JsonResponse(response)

@api_view(['GET'])
@csrf_exempt
def lookup_svm(request, dataset_name):
    label_function = request.GET['user']
    label_category = request.GET['category']
    cluster_id = request.GET['cluster_id']
    index_id = request.GET['index_id']
    augmentations = [x.split(":") for x in request.GET['augmentations'].split(',')]
    use_full_image = bool(distutils.util.strtobool(request.GET['use_full_image']))
    mode = request.GET['mode']
    prev_svm_vector = request.GET['prev_svm_vector']
    autolabel_percent = float(request.GET['autolabel_percent'])
    autolabel_max_vectors = int(request.GET['autolabel_max_vectors'])

    # 1. Retrieve dataset info from db
    dataset = Dataset.objects.get(name=dataset_name)
    data_directory = dataset.directory
    split_dir = data_directory[len('gs://'):].split('/')
    bucket_name = split_dir[0]
    bucket_path = '/'.join(split_dir[1:])

    # Get positive paths, negative paths, positive bounding boxes
    # getNextImages shows good way to do this
    # Positive patches:
    pos_anns = Annotation.objects.filter(
        Q(label_type="klabel_extreme") | Q(label_type="klabel_box"),
        dataset_item__in=dataset.datasetitem_set.filter(),
        label_function=label_function,
        label_category=label_category)
    pos_paths = [ann.dataset_item.path for ann in pos_anns]
    pos_patches = [{'x1': bbox['bmin']['x'],
                'y1': bbox['bmin']['y'],
                'x2': bbox['bmax']['x'],
                'y2': bbox['bmax']['y']}
               for bbox in [json.loads(ann.label_data)['bbox']
                            for ann in pos_anns]]
    frame_anns = Annotation.objects.filter(
        dataset_item__in=dataset.datasetitem_set.filter(),
        label_type="klabel_frame",
        label_function=label_function,
        label_category=label_category)
    neg_paths = []
    for ann in frame_anns:
        label_value = json.loads(ann.label_data)['value']
        if label_value in (
            LabelType.negative.value,
            LabelType.hard_negative.value,
        ):
            neg_paths.append(ann.dataset_item.path)

    # 3. Send paths and patches to /query_svm
    params = {
        "cluster_id": cluster_id,
        "index_id": index_id,
        "bucket": bucket_name,
        "positive_paths": pos_paths,
        "positive_patches": pos_patches,
        "negative_paths": neg_paths,
        "augmentations": augmentations,
        "num_results": 100,
        "use_full_image": use_full_image,
        "mode": mode,
        "prev_svm_vector": prev_svm_vector,
        "autolabel_percent": autolabel_percent,
        "autolabel_max_vectors": autolabel_max_vectors,
        "log_id_string": "logid:[{:s}:{:s}]".format(label_function, label_category)
    }
    print(params)
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_svm",
        json=params,
    )
    response_data = r.json()

    response = process_image_query_results(request, dataset, response_data)

    # 4. Return knn results
    return JsonResponse(response)

@api_view(['GET'])
@csrf_exempt
def active_batch(request, dataset_name):
    label_function = request.GET['user']
    label_category = request.GET['category']
    cluster_id = request.GET['cluster_id']
    index_id = request.GET['index_id']
    augmentations = [x.split(":") for x in request.GET['augmentations'].split(',')]

    # 1. Retrieve dataset info from db
    dataset = Dataset.objects.get(name=dataset_name)
    data_directory = dataset.directory
    split_dir = data_directory[len('gs://'):].split('/')
    bucket_name = split_dir[0]
    bucket_path = '/'.join(split_dir[1:])

    # 2. Get google images
    _, paths, _ = query_google(request, dataset)
    print(paths)

    # 3. Send paths and patches to /active_batch
    params = {
        "cluster_id": cluster_id,
        "index_id": index_id,
        "bucket": bucket_name,
        "paths": paths,
        "augmentations": augmentations,
        "num_results": 100
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/active_batch",
        json=params,
    )
    response_data = r.json()

    # Apply filtering to generated order
    response = process_image_query_results(request, dataset, response_data)

    # 4. Return knn results
    return JsonResponse(response)


#
# V2 ENDPOINTS
#


def filtered_images_v2(request, dataset, path_filter=None):
    include_categories = {c for c in request.GET.get("include").split(",") if c}
    exclude_categories = {c for c in request.GET.get("exclude").split(",") if c}

    path_filter_kwargs = {"path__in": path_filter} if path_filter else {}
    dataset_items = DatasetItem.objects.filter(
        dataset=dataset, google=False, **path_filter_kwargs
    ).order_by("pk")

    if not include_categories and not exclude_categories:
        return dataset_items

    # TODO(mihirg, fpoms): Speed up by filtering positive labels without having
    # to json decode all annotations
    annotations = Annotation.objects.filter(
        dataset_item__in=dataset_items,
        label_category__in=list(
            itertools.chain(include_categories, exclude_categories)
        ),
        label_type="klabel_frame"
    )
    anns = filter_most_recent_anns(
        nest_anns(annotations, nest_lf=False)
    )  # [image][category][#]

    pks_to_return = set()
    include_categories = set(include_categories)
    exclude_categories = set(exclude_categories)
    for di_pk, anns_by_cat in anns.items():
        include = False
        exclude = False
        for cat, ann_list in anns_by_cat.items():
            for ann in ann_list:
                label_value = json.loads(ann.label_data)
                if label_value["value"] != LabelType.positive.value:
                    continue

                if cat in include_categories:
                    include = True
                else:  # cat in exclude_categories
                    exclude = True
                    break

            if exclude:
                break

        if include and not exclude:
            pks_to_return.add(di_pk)

    return [di for di in dataset_items if di.pk in pks_to_return]


@api_view(["GET"])
@csrf_exempt
def query_knn_v2(request, dataset_name):
    index_id = request.GET["index_id"]
    image_ids = request.GET["image_ids"].split(",")
    num_results = int(request.GET.get("num", 1000))

    dataset = Dataset.objects.get(name=dataset_name)
    dataset_items = DatasetItem.objects.filter(pk__in=image_ids)
    dataset_item_identifiers = [di.identifier for di in dataset_items]

    params = {
        "index_id": index_id,
        "num_results": num_results,
        "identifiers": dataset_item_identifiers,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_knn_v2",
        json=params,
    )
    response_data = r.json()
    response = process_image_query_results(
        request,
        dataset,
        response_data,
        filtered_images_v2
    )
    return JsonResponse(response)


@api_view(["GET"])
@csrf_exempt
def get_next_images_v2(request, dataset_name, dataset=None):
    if not dataset:
        dataset = get_object_or_404(Dataset, name=dataset_name)

    bucket_name = dataset.directory[len("gs://"):].split("/")[0]
    path_template = "https://storage.googleapis.com/{:s}/".format(bucket_name) + "{:s}"
    offset_to_return = int(request.GET.get("offset", 0))
    num_to_return = int(request.GET.get("num", 1000))
    index_id = request.GET["index_id"]

    all_images = filtered_images_v2(request, dataset)
    ret_images = all_images[
        offset_to_return : offset_to_return + num_to_return
    ]

    identifiers = [di.identifier for di in ret_images]
    params = {
        "index_id": index_id,
        "identifiers": identifiers,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/perform_clustering",
        json=params,
    )
    clustering_data = r.json()

    dataset_item_paths = [
        (di.path if di.path.find("http") != -1 else path_template.format(di.path))
        for di in ret_images
    ]
    dataset_item_identifiers = [di.pk for di in ret_images]

    return JsonResponse({
        "paths": dataset_item_paths,
        "identifiers": dataset_item_identifiers,
        "num_total": len(all_images),
        "clustering": clustering_data["clustering"],
    })


# TODO(mihirg): Make this faster
@api_view(["GET"])
@csrf_exempt
def get_dataset_info_v2(request, dataset_name):
    dataset = Dataset.objects.get(name=dataset_name)

    annotations = Annotation.objects.filter(
        dataset_item__in=dataset.datasetitem_set.filter()
    )
    categories = set()
    for ann in annotations:
        categories.add(ann.label_category)
    categories = sorted(list(categories))

    return JsonResponse({
        "categories": categories,
        "index_id": dataset.index_id,
        "num_images": dataset.datasetitem_set.filter(google=False).count(),
        "num_google": dataset.datasetitem_set.filter(google=True).count()
    })
