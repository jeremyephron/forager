import json
import os
import http.client
import urllib
import json
import time
import requests

from django.http import HttpRequest, HttpResponse, JsonResponse
from google.cloud import storage
from django.core.exceptions import ValidationError
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import  get_object_or_404, get_list_or_404
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from collections import defaultdict

from .models import Dataset, DatasetItem, Annotation

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
def create_dataset(request):
    try:
        data = json.loads(request.body)
        name = data['dataset']
        data_directory = data['dirpath']
        if not data_directory.startswith('gs://'):
            err = ValidationError(
                'Directory only supports Google Storage bucket paths. '
                'Please specify as "gs://bucket-name/path/to/data".')
            # form.add_error('directory', err)
            raise err

        split_dir = data_directory[len('gs://'):].split('/')
        bucket_name = split_dir[0]
        bucket_path = '/'.join(split_dir[1:])

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        all_blobs = client.list_blobs(bucket, prefix=bucket_path)
        # all_blobs = []

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

        # Start background job to start processing dataset
        embedding_conn = http.client.HTTPConnection(
            settings.EMBEDDING_SERVER_ADDRESS)
        if 'cluster_id' in request.session:
            # Check if stil exists
            params = {'cluster_id': request.session['cluster_id']}
            r = requests.get(
                settings.EMBEDDING_SERVER_ADDRESS + '/cluster_status',
                params=params
            )
            response_data = r.json()
            if not response_data['has_cluster']:
                del request.session['cluster_id']
        if not 'cluster_id' in request.session:
            # Create cluster if it does not exist
            headers = {"Content-type": "application/x-www-form-urlencoded",
                       "Accept": "application/json"}
            params = {'n_nodes': settings.EMBEDDING_CLUSTER_NODES}
            r = requests.post(
                settings.EMBEDDING_SERVER_ADDRESS + '/start_cluster',
                data=params, headers=headers)
            response_data = r.json()
            request.session['cluster_id'] = response_data['cluster_id']
        # Initiate embedding computation
        headers = {"Content-type": "application/x-www-form-urlencoded",
                   "Accept": "application/json"}
        params = {
            'cluster_id': request.session['cluster_id'],
            'n_mappers': settings.EMBEDDING_MAPPERS,
            'bucket': bucket_name,
            'paths': paths}
        r = requests.post(
            settings.EMBEDDING_SERVER_ADDRESS + '/start',
            data=params, headers=headers)
        response_data = r.json()

        query_id = response_data['query_id']

        # Track status of the embedding computation
        value_at_last = None
        same_as_last = 0
        SAME_AS_ATTEMPTS = 10
        while True:
            params = {'query_id': query_id}
            r = requests.get(
                settings.EMBEDDING_SERVER_ADDRESS + '/results',
                params=params
            )
            response_data = r.json()

            if 'progress' in response_data:
                n_processed = response_data['progress']['n_processed']
                n_total = response_data['progress']['n_total']
                n_skipped = response_data['progress']['n_skipped']
                print('{:d}/{:d} ({:d} skipped)'.format(
                    n_processed, n_total, n_skipped))
                if n_processed == value_at_last and n_processed != 0:
                    same_as_last += 1
                    if same_as_last == SAME_AS_ATTEMPTS:
                        break
                else:
                    value_at_last = n_processed
                    same_as_last = 0
            if response_data['progress']['finished']:
                break
            time.sleep(5)

        # Create index using the embedding results
        headers = {"Content-type": "application/x-www-form-urlencoded",
                   "Accept": "application/json"}
        params = {"query_id": query_id}
        r = requests.post(
            settings.EMBEDDING_SERVER_ADDRESS + "/create_index",
            data=params, headers=headers
        )
        response_data = r.json()
        request.session["index_id"] = response_data["index_id"]

        # return HttpResponse(json.dumps({'status': 'success'}), content_type='application/json')
        req = HttpRequest()
        req.method = 'GET'
        return get_dataset_info(req, name, dataset)
    except ValidationError as e:
        # Redisplay the question voting form.
        return JsonResponse({
            'status': 'failure',
            'message': ('Something went wrong. Make sure the directory path is '
                       'valid.')
        })


@api_view(['GET'])
@csrf_exempt
def get_dataset_info(request, dataset_name, dataset=None):
    if not dataset:
        dataset = get_object_or_404(Dataset, name=dataset_name)

    filter_args = dict(dataset=dataset)

    bucket_name = dataset.directory[len('gs://'):].split('/')[0]
    path_template = 'https://storage.googleapis.com/{:s}/'.format(bucket_name) + '{:s}'
    dataset_items = DatasetItem.objects.filter()[:100]
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
def get_annotation_conflicts(request, dataset_name):
    image_identifiers = request.GET['identifiers'].split(',')
    label_function = request.GET['user']
    category = request.GET['category']

    filter_args = dict(
        dataset_item__in=dataset_items,
        label_type='klabel_perframe'
    )
    if not category == 'all':
        filter_args['label_category'] = category

    dataset_items = DatasetItem.objects.filter(pk__in=image_identifiers)
    anns = Annotation.objects.filter(**filter_args)

    data = defaultdict(list)
    anns_by_image = defaultdict(list)
    for ann in anns:
        label_data = json.loads(ann.label_data)
        label_data['identifier'] = ann.pk
        data[ann.dataset_item.pk].append(label_data)

    # Analyze conflicts
    conflict_data = defaultdict(list)
    for image_id, labels in data.items():
        has_user_label = False
        user_annotation = None
        for ann in labels:
            if ann.label_function == label_function:
                has_user_label = True
                user_annotation = ann
                break
        if not has_user_label:
            continue
        for ann in labels:
            if ann == user_annotation:
                continue
            if ann.label_data != user_annotation.label_data:
                conflict_data[image_id] = labels
                break

    return JsonResponse(conflict_data)


@api_view(['POST'])
@csrf_exempt
def add_annotation(request, dataset_name, image_identifier):
    # body is the JSON of a single annotation
    # {
    #   'type': 2,
    #   'bbox': {'bmin': {'x': 0.2, 'y': 0.5}, 'bmax': {'x': 0.5, 'y': 0.7}}
    # }
    body = request.body.decode('utf-8')
    label_function = body['user']
    category = body['category']
    annotation = body['annotation']

    dataset_item = DatasetItem.objects.get(pk=image_identifier)
    ann = Annotation(
        dataset_item=dataset_item,
        label_function=label_function,
        label_category=category,
        label_type='klabel',
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
    # NOTE(fpoms): embedding computation does not support black/white images rn.
    cluster_id = request.session['cluster_id']
    index_id = request.session['index_id']
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "application/json"}
    params = {
        "cluster_id": cluster_id,
        "index_id": index_id,
        "bucket": bucket_name,
        "paths": paths,
        "patches": json.dumps(patches),
        "num_results": 100,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_index",
        data=params, headers=headers
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
