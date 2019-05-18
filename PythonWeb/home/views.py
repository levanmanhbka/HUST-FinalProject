from django.shortcuts import render
from django.http import HttpResponse
from .models import Lanmark
from django.views.generic import ListView, DetailView
# Create your views here.
def index(request):
    lanmarks =  {"lanmarks":Lanmark.objects.all().order_by("num")}
    return render(request, "pages/home.html", lanmarks)

def detail(request, id):
    lanmark = Lanmark.objects.get(id=id)
    print(id)
    return render(request, "pages/detail.html", {'lanmark':lanmark})

class IndexView(ListView):
    queryset = Lanmark.objects.all().order_by("num")
    template_name = "pages/home.html"
    context_object_name = "lanmarks"
    paginate_by = 12

class LanmarkDetail(DetailView):
    model = Lanmark
    template_name = "pages/detail.html"
    context_object_name = "lanmark"