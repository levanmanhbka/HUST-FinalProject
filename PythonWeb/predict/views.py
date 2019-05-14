from django.shortcuts import render
from .models import Post
# Create your views here.

def list(request):
    data =  {"posts":Post.objects.all().order_by("-date")}
    return render(request, "predict/predict.html", data)
