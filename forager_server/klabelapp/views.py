from django.shortcuts import render
from .forms import CreateDatasetForm

# Create your views here.

def index(request):
    return render(request, 'klabelapp/index.html')


def new_dataset(request):
    form = CreateDatasetForm(request.POST, request.FILES)
    return render(request, 'klabelapp/new_dataset.html', {'form': form})


def do_new_dataset(request):
    form = CreateDatasetForm(request.POST, request.FILES)
    try:
        if not form.is_valid():
            raise ValidationError('')
        return HttpResponseRedirect(reverse('klabelapp:dataset') + "?created=true")
    except ValidationError:
        # Redisplay the question voting form.
        return render(request, 'klabelapp/new_dataset.html', {'form': form})


def dataset(request):
    return render(request, 'klabelapp/dataset.html')


def label(request):
    return render(request, 'klabelapp/klabel.html')
