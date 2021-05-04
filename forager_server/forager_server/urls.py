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
    path('api/start_cluster',
         views.start_cluster,
         name='start_cluster'),
    path('api/cluster/<slug:cluster_id>',
         views.get_cluster_status,
         name='get_cluster_status'),
    path('api/stop_cluster/<slug:cluster_id>',
         views.stop_cluster,
         name='stop_cluster'),
    path('api/get_results_v2/<slug:dataset_name>',
         views.get_results_v2,
         name='get_results_v2'),
    path('api/generate_embedding_v2',
         views.generate_embedding_v2,
         name='generate_embedding_v2'),
    path('api/generate_text_embedding_v2',
         views.generate_text_embedding_v2,
         name='generate_text_embedding_v2'),
    path('api/query_knn_v2/<slug:dataset_name>',
         views.query_knn_v2,
         name='query_knn_v2'),
    path('api/train_svm_v2/<slug:dataset_name>',
         views.train_svm_v2,
         name='train_svm_v2'),
    path('api/query_svm_v2/<slug:dataset_name>',
         views.query_svm_v2,
         name='query_svm_v2'),
    path('api/query_ranking_v2/<slug:dataset_name>',
         views.query_ranking_v2,
         name='query_ranking_v2'),
    path('api/query_images_v2/<slug:dataset_name>',
         views.query_images_v2,
         name='query_images_v2'),
    path('api/query_metrics_v2/<slug:dataset_name>',
         views.query_metrics_v2,
         name='query_metrics_v2'),
    path('api/query_active_validation_v2/<slug:dataset_name>',
         views.query_active_validation_v2,
         name='query_active_validation_v2'),
    path('api/add_val_annotations_v2',
         views.add_val_annotations_v2,
         name='add_val_annotations_v2'),
    path('api/get_dataset_info_v2/<slug:dataset_name>',
         views.get_dataset_info_v2,
         name='get_dataset_info_v2'),
    path('api/get_models_v2/<slug:dataset_name>',
         views.get_models_v2,
         name='get_models_v2'),
    path('api/get_annotations_v2',
         views.get_annotations_v2,
         name='get_annotations_v2'),
    path('api/add_annotations_v2',
         views.add_annotations_v2,
         name='add_annotations_v2'),
    path('api/add_annotations_to_result_set_v2',
         views.add_annotations_to_result_set_v2,
         name='add_annotations_to_result_set_v2'),
    path('api/delete_category_v2',
         views.delete_category_v2,
         name='delete_category_v2'),
    path('api/update_category_v2',
         views.update_category_v2,
         name='update_category_v2'),
    path('api/get_category_counts_v2/<slug:dataset_name>',
         views.get_category_counts_v2,
         name='get_category_counts_v2'),
    path('api/train_model_v2/<slug:dataset_name>',
         views.create_model,
         name='create_model'),
    path('api/model_v2/<slug:model_id>',
         views.get_model_status,
         name='get_model_status'),
    path('api/model_inference_v2/<slug:dataset_name>',
         views.run_model_inference,
         name='run_model_inference'),
    path('api/model_inference_status_v2/<slug:job_id>',
         views.get_model_inference_status,
         name='get_model_inference_status'),
    path('api/stop_model_inference_v2/<slug:job_id>',
         views.stop_model_inference,
         name='stop_model_inference'),
]
