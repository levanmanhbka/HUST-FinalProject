from django.urls import path
from .views import MyView
views = MyView()
urlpatterns = [
    path('', views.index, name = 'home'),
    path('<int:id>', views.detail, name = 'detail'),
    path('upload', views.upload_file, name='upload_file')
]