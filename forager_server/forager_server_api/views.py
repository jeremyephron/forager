import json
import os
import requests
import time

from django.http import HttpRequest, HttpResponse, JsonResponse
from google.cloud import storage
from django.core.exceptions import ValidationError, ObjectDoesNotExist
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_object_or_404
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from collections import defaultdict

from .models import Dataset, DatasetItem, Annotation


POST_HEADERS = {
    "Content-type": "application/x-www-form-urlencoded",
    "Accept": "application/json",
}

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

    if isinstance(next(iter(nested_anns.items()))[1], list):
       data = defaultdict(list)
       for pk, anns in nested_anns.items():
           data[pk] = filter_fn(anns)
    if isinstance(next(iter(next(iter(nested_anns.items()))[1].items())), list):
       data = defaultdict(lambda: defaultdict(list))
       for pk, label_fns_data in nested_anns.items():
           for label_fn, anns in label_fns_data.items():
               data[pk][label_fn] = filter_fn(anns)
    if isinstance(next(iter(next(iter(next(iter(nested_anns.items()))[1].items()))[1].items()))[1], list):
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
    params = {"n_nodes": settings.EMBEDDING_CLUSTER_NODES}
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/start_cluster",
        data=params,
        headers=POST_HEADERS,
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
        data=params,
        headers=POST_HEADERS,
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
    dataset_items = DatasetItem.objects.filter(dataset=dataset)
    dataset_item_raw_paths = [di.path for di in dataset_items]

    data = json.loads(request.body)
    params = {
        "cluster_id": data["cluster_id"],
        "bucket": bucket_name,
        "paths": dataset_item_raw_paths,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/start_job",
        data=params,
        headers=POST_HEADERS,
    )
    response_data = r.json()
    return JsonResponse({
        "status": "success",
        "index_id": response_data["index_id"],
    })


@api_view(['GET'])
@csrf_exempt
def get_index_status(request, index_id):
    params = {"index_id": index_id}
    r = requests.get(
        settings.EMBEDDING_SERVER_ADDRESS + "/job_status", params=params
    )
    response_data = r.json()
    return JsonResponse(response_data)


@api_view(['POST'])
@csrf_exempt
def delete_index(request, index_id):
    params = {"index_id": index_id}
    requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/delete_index",
        data=params,
        headers=POST_HEADERS,
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
        items = [
            DatasetItem(dataset=dataset,
                        identifier=os.path.basename(path).split('.')[0],
                        path=path)
            for path in paths
        ]
        DatasetItem.objects.bulk_create(items)

        req = HttpRequest()
        req.method = 'GET'
        return get_dataset_info(req, name, dataset)
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
    dataset_items = DatasetItem.objects.filter(dataset=dataset).order_by('pk')[:100]
    dataset_item_paths = [path_template.format(di.path) for di in dataset_items]
    dataset_item_identifiers = [di.pk for di in dataset_items]

    return JsonResponse({
        'status': 'success',
        'datasetName': dataset.name,
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


@api_view(['GET'])
@csrf_exempt
def get_next_images(request, dataset_name, dataset=None):
    if not dataset:
        dataset = get_object_or_404(Dataset, name=dataset_name)

    CATEGORIES = {
        'positive': 1,
        'negative': 2,
        'hard_negative': 3,
        'unsure': 4,
    }
    label_function = request.GET['user']
    category = request.GET['category']
    label_value = request.GET['filter']
    offset_to_return = int(request.GET.get('offset', 0))
    num_to_return = int(request.GET.get('num', 100))

    bucket_name = dataset.directory[len('gs://'):].split('/')[0]
    path_template = 'https://storage.googleapis.com/{:s}/'.format(bucket_name) + '{:s}'
    dataset_items = DatasetItem.objects.filter(dataset=dataset).order_by('pk')

    # Find all annotations for the user and category in this dataset
    annotations = Annotation.objects.filter(
        dataset_item__in=dataset_items,
        label_function=label_function,
        label_category=category,
        label_type='klabel_frame')

    next_images = []
    if label_value == 'all':
        next_images = dataset_items
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
        cat_value = CATEGORIES[label_value]
        images_with_label = set()
        for ann in annotations:
            label_value = json.loads(ann.label_data)
            if label_value['value'] != cat_value:
                continue
            images_with_label.add(ann.dataset_item)
        for ditem in dataset_items:
            if ditem in images_with_label:
                next_images.append(ditem)

    ret_images = next_images[offset_to_return:offset_to_return+num_to_return]

    # Find all dataset items which do not have an annotation of the type
    dataset_item_paths = [path_template.format(di.path) for di in ret_images]
    dataset_item_identifiers = [di.pk for di in ret_images]

    return JsonResponse({
        'paths': dataset_item_paths,
        'identifiers': dataset_item_identifiers,
        'num_total': len(next_images),
    })


@api_view(['GET'])
@csrf_exempt
def get_results(request, dataset_name):
    # Placeholder
    dataset = get_object_or_404(Dataset, name=dataset_name)
    bucket_name = dataset.directory[len('gs://'):].split('/')[0]
    path_template = 'https://storage.googleapis.com/{:s}/'.format(bucket_name) + '{:s}'
    dataset_items = DatasetItem.objects.filter(dataset=dataset)[:100]
    dataset_item_paths = [path_template.format(di.path) for di in dataset_items]
    dataset_item_identifiers = [di.pk for di in dataset_items]

    return JsonResponse([
        {'path': p, 'idx': i}
        for i, p in list(enumerate(dataset_item_paths))[50:100]
    ], safe=False)


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
    total_images = dataset.datasetitem_set.count()
    anns = Annotation.objects.filter(
        label_type="klabel_frame",
        dataset_item__in=dataset.datasetitem_set.filter())
    label_functions = [d['label_function'] for d in (anns.values("label_function").distinct())]
    label_categories = [d['label_category'] for d in (anns.values("label_category").distinct())]

    print(label_functions)
    print(label_categories)
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
    total_images = dataset.datasetitem_set.count()
    anns = Annotation.objects.filter(
        dataset_item__in=dataset.datasetitem_set.filter())
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
        user_annotation = user_labels[label_function]
        user_label_value = json.loads(user_annotation.label_data)
        for label_function, ann in user_labels.items():
            if label_function == user_annotation.label_function:
                continue
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


@api_view(['GET'])
@csrf_exempt
def lookup_knn(request, dataset_name):
    ann_identifiers = [int(x) for x in request.GET['ann_identifiers'].split(',')]
    cluster_id = request.GET['cluster_id']
    index_id = request.GET['index_id']
    augmentations = [x.split(":") for x in request.GET['augmentations'].split(',')]
    print("Augmentations: ")
    print(augmentations)
    use_full_image = 'use_full_image' in request.GET

    # 1. Retrieve dataset info from db
    dataset = Dataset.objects.get(name=dataset_name)
    data_directory = dataset.directory
    split_dir = data_directory[len('gs://'):].split('/')
    bucket_name = split_dir[0]
    bucket_path = '/'.join(split_dir[1:])

    # 2. Retrieve annotations from db
    query_annotations = Annotation.objects.filter(pk__in=ann_identifiers)
    paths = [ann.dataset_item.path for ann in query_annotations]
    patches = [{'x1': bbox['bmin']['x'],
                'y1': bbox['bmin']['y'],
                'x2': bbox['bmax']['x'],
                'y2': bbox['bmax']['y']}
               for bbox in [json.loads(ann.label_data)['bbox']
                            for ann in query_annotations]]

    # 3. Send paths and patches to /query_index
    params = {
        "cluster_id": cluster_id,
        "index_id": index_id,
        "bucket": bucket_name,
        "paths": paths,
        "patches": json.dumps(patches),
        "augmentations": augmentations,
        "num_results": 100,
        "use_full_image": use_full_image,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_index",
        data=params, headers=POST_HEADERS
    )
    response_data = r.json()
    response = {
        'identifiers': [],
        'paths': []
    }
    ditems = DatasetItem.objects.filter(path__in=response_data['results'])
    path_to_id = {}
    for ditem in ditems:
        path_to_id[ditem.path] = ditem.pk
    for path in response_data['results']:
        path_template = 'https://storage.googleapis.com/{:s}/'.format(bucket_name) + '{:s}'
        response['paths'].append(path_template.format(path))
        response['identifiers'].append(path_to_id[path])

    # 4. Return knn results
    return JsonResponse(response)
