from django.urls import path

from . import views

app_name='klabelapp'
urlpatterns = [
    path('', views.index, name='index'),
    path('new_dataset', views.new_dataset, name='new_dataset'),
    path('do_new_dataset', views.do_new_dataset, name='do_new_dataset'),
    path('dataset/<slug:dataset_name>/label/', views.label, name='label'),
]
