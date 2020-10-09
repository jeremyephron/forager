"""forager_server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from forager_server_api import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/get_datasets',
         views.get_datasets,
         name='get_datasets'),
    path('api/create_dataset',
         views.create_dataset,
         name='create_dataset'),
    path('api/dataset/<slug:dataset_name>',
         views.get_dataset_info,
         name='get_dataset_info'),
    path('api/get_results/<slug:dataset_name>',
         views.get_results,
         name='get_results'),
    path('api/get_annotations/<slug:dataset_name>',
         views.get_annotations,
         name='get_annotations'),
    path('api/add_annotation/<slug:dataset_name>/<slug:image_identifier>',
         views.add_annotation,
         name='add_annotation'),
    path('api/delete_annotation/<slug:dataset_name>/<slug:image_identifier>/<slug:ann_identifier>',
         views.delete_annotation,
         name='delete_annotation'),
    path('api/lookup_knn/<slug:dataset_name>',
         views.lookup_knn,
         name='lookup_knn'),
]
