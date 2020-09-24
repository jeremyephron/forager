from django.shortcuts import render
from django.http import HttpResponseRedirect
from google.cloud import storage
from django.core.exceptions import ValidationError
from django.urls import reverse
from django.shortcuts import render, get_object_or_404, get_list_or_404

from .forms import CreateDatasetForm
from .models import Dataset

# Create your views here.
def index(request):
    datasets = Dataset.objects.filter()
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
        bucket_path = split_dir[1:]

        #client = storage.Client()
        #bucket = storage.Bucket(bucket_name)

        # Create all the DatasetItems for this dataset

        # Start background job to start processing dataset

        dataset = Dataset(name=name, directory=data_directory)
        dataset.save()

        return HttpResponseRedirect(reverse('klabelapp:index') + "?created=true")
    except ValidationError:
        # Redisplay the question voting form.
        return render(request, 'klabelapp/new_dataset.html', {'form': form})


def dataset(request):
    return render(request, 'klabelapp/dataset.html')


def label(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)
    return render(request, 'klabelapp/klabel.html', {'dataset': dataset})
