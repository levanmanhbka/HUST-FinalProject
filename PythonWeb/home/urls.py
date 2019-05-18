from django.urls import path
from .import views

urlpatterns = [
    path('', views.IndexView.as_view(), name = 'home'),
    path('<int:pk>', views.LanmarkDetail.as_view(), name = 'detail')
]