from django.shortcuts import render
from django.http import HttpResponse
from .models import Lanmark
# Create your views here.
def index(request):
    lanmarks =  {"lanmarks":Lanmark.objects.all().order_by("num")}
    return render(request, "pages/home.html", lanmarks)
