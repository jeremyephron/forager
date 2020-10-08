import os.path

from django.shortcuts import render
from django.http import HttpResponseRedirect
from google.cloud import storage
from django.core.exceptions import ValidationError
from django.urls import reverse
from django.shortcuts import render, get_object_or_404, get_list_or_404

from .forms import CreateDatasetForm
from .models import Dataset, DatasetItem
from .tasks import generate_embeddings_task

# Create your views here.
def index(request):
    datasets = Dataset.objects.filter()
    for dataset in datasets:
        print(dir(dataset))
    return render(request, 'klabelapp/index.html', {'datasets': datasets})


def new_dataset(request):
    form = CreateDatasetForm()
    return render(request, 'klabelapp/new_dataset.html', {'form': form})


def do_new_dataset(request):
    form = CreateDatasetForm(request.POST, request.FILES)
    try:
        if not form.is_valid():
            raise ValidationError('')

        name = form.cleaned_data['name']
        data_directory = form.cleaned_data['directory']
        if not data_directory.startswith('gs://'):
            err = ValidationError(
                'Directory only supports Google Storage bucket paths. '
                'Please specify as "gs://bucket-name/path/to/data".')
            form.add_error('directory', err)
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
        print('num paths', len(paths))
        DatasetItem.objects.bulk_create(items)

        # Start background job to start processing dataset
        generate_embeddings_task(name)
        # TODO(fpoms): implement background job

        return HttpResponseRedirect(reverse('klabelapp:index') + "?created=true")
    except ValidationError:
        # Redisplay the question voting form.
        return render(request, 'klabelapp/new_dataset.html', {'form': form})


def dataset(request):
    return render(request, 'klabelapp/dataset.html')


def label(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)

    bucket_name = dataset.directory[len('gs://'):].split('/')[0]
    path_template = '"https://storage.googleapis.com/{:s}/'.format(bucket_name) + '{:s}"'
    dataset_items = DatasetItem.objects.filter(dataset=dataset)[:100]
    dataset_item_paths = [di.path for di in dataset_items]
    dataset_item_paths_str = '[{:s}]'.format(
        ','.join(
            [path_template.format(path) for path in dataset_item_paths]))
    dataset_item_identifiers = [di.identifier for di in dataset_items]
    dataset_item_identifiers_str = '[{:s}]'.format(
        ','.join(
            ['"{:s}"'.format(path) for path in dataset_item_identifiers]))
    return render(request, 'klabelapp/klabel.html',
                  {'dataset': dataset,
                   'paths_str': dataset_item_paths_str,
                   'identifiers_str': dataset_item_identifiers_str})
