from django.shortcuts import render
from django.http import HttpResponse
from .models import Lanmark
from django.views.generic import ListView, DetailView
from django.http import HttpResponseRedirect
from django.core.files.storage import FileSystemStorage
from .reoder_lanmark import ReoderLanmark

# Create your views here.
def index(request, uploaded= "default"):
    print("uploaded= ", uploaded)
    lanmarks =  {"lanmarks":Lanmark.objects.all().order_by("num"), "uploaded":uploaded}
    return render(request, "pages/home.html", lanmarks)

def detail(request, id):
    lanmark = Lanmark.objects.get(id=id)
    print(id)
    return render(request, "pages/detail.html", {'lanmark':lanmark})

class IndexView(ListView):
    queryset = Lanmark.objects.all().order_by("num")
    template_name = "pages/home.html"
    context_object_name = "lanmarks"
    paginate_by = 100

class LanmarkDetail(DetailView):
    model = Lanmark
    template_name = "pages/detail.html"
    context_object_name = "lanmark"

def upload_file(request):
    if request.method == 'POST':
        myfile = request.FILES['myfile']
        print("file name= ", myfile.name)
        print("file size= ", myfile.size)
        # Save image request file
        fs = FileSystemStorage()
        fs.save("uploaded/" + str(myfile.name), myfile)
        # Use machine learning to reoder list post
        reoder = ReoderLanmark()
        reoder.get_new_order(str(fs.location) + str("/uploaded/") + str(myfile.name))
    return index(request, str("uploaded/") + str(myfile.name))