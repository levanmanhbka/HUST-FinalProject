from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
def index(request):
    images = ["images/nature.jpg", "images/nature.jpg", "images/nature.jpg", "images/nature.jpg", "images/nature.jpg"]
    return render(request, "pages/home.html", {'images':images})
