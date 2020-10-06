import json
import os

from django.http import HttpResponse
from google.cloud import storage
from django.core.exceptions import ValidationError
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import  get_object_or_404, get_list_or_404

from .models import Dataset, DatasetItem


def get_datasets(request):
    datasets = Dataset.objects.filter()

    def serialize(dataset):
        # JEB: probably should make a real serializer
        n_labels = 0
        if getattr(dataset.datasetitem_set, 'label_set', None):
            n_labels = dataset.datasetitem_set.label_set.count()

        return {
            'name': dataset.name, 
            'size': dataset.datasetitem_set.count(),
            'n_labels': n_labels,
            'last_labeled': 'N/A'
        }

    payload = json.dumps(list(map(serialize, datasets)))
    return HttpResponse(payload, content_type='application/json')


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

        dataset = Dataset(name=name, directory=data_directory)
        dataset.save()

        # Create all the DatasetItems for this dataset
        paths = [blob.name for blob in all_blobs]
        items = [
            DatasetItem(dataset=dataset, identifier=os.path.basename(path).split('.')[0], path=path)
            for path in paths
        ]
        DatasetItem.objects.bulk_create(items)
        print('SUCCESS')

        # Start background job to start processing dataset
        # TODO(fpoms): implement background job

        # return HttpResponse(json.dumps({'status': 'success'}), content_type='application/json')
        return get_dataset_info(None, name, dataset)
    except ValidationError as e:
        # Redisplay the question voting form.
        return HttpResponse(json.dumps({
            'status': 'failure',
            'message': ('Something went wrong. Make sure the directory path is '
                       'valid.')
        }), content_type='application/json')


def get_dataset_info(request, dataset_name, dataset=None):
    if not dataset:
        dataset = get_object_or_404(Dataset, name=dataset_name)

    bucket_name = dataset.directory[len('gs://'):].split('/')[0]
    path_template = 'https://storage.googleapis.com/{:s}/'.format(bucket_name) + '{:s}'
    dataset_items = DatasetItem.objects.filter(dataset=dataset)[:100]
    dataset_item_paths = [path_template.format(di.path) for di in dataset_items]
    dataset_item_identifiers = [di.identifier for di in dataset_items]

    return HttpResponse(json.dumps({
        'status': 'success',
        'datasetName': dataset.name,
        'paths': dataset_item_paths,
        'identifiers': dataset_item_identifiers
    }), content_type='application/json')


def get_results(request, dataset_name):
    # Placeholder
    dataset = get_object_or_404(Dataset, name=dataset_name)
    bucket_name = dataset.directory[len('gs://'):].split('/')[0]
    path_template = 'https://storage.googleapis.com/{:s}/'.format(bucket_name) + '{:s}'
    dataset_items = DatasetItem.objects.filter(dataset=dataset)[:100]
    dataset_item_paths = [path_template.format(di.path) for di in dataset_items]
    dataset_item_identifiers = [di.identifier for di in dataset_items]

    return HttpResponse(json.dumps([
        {'path': p, 'idx': i} for i, p in list(enumerate(dataset_item_paths))[50:100]
    ]), content_type='application/json')
