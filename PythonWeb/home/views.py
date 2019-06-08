from django.shortcuts import render
from django.http import HttpResponse
from .models import Lanmark
from django.views.generic import ListView, DetailView
from django.http import HttpResponseRedirect
from django.core.files.storage import FileSystemStorage
from .reoder_lanmark import ReoderLanmark

class MyView():
    def __init__(self):
        self.list_index = [i for i in range(48)]
        print(self.list_index)
    
    def sort_query_by_index(self, list_query):
        ""
        list_lanscape = []
        for index in self.list_index:
            for element in list_query:
                if element.num == index:
                    list_lanscape.append(element)
                    break
        return list_lanscape

    # Create your views here.
    def index(self,request, uploaded= "default"):
        print("uploaded= ", uploaded)
        list_lanscape = Lanmark.objects.all().order_by("num")
        print("default list: ",list_lanscape)
        if uploaded != "default":
            list_lanscape = self.sort_query_by_index(list_lanscape)
            print("sorted list: ", list_lanscape)
        lanmarks =  {"lanmarks":list_lanscape, "uploaded":uploaded}
        return render(request, "pages/home.html", lanmarks)

    def detail(self,request, id):
        lanmark = Lanmark.objects.get(id=id)
        print(id)
        return render(request, "pages/detail.html", {'lanmark':lanmark})

    def upload_file(self,request):
        if request.method == 'POST':
            myfile = request.FILES['myfile']
            print("file name= ", myfile.name)
            print("file size= ", myfile.size)
            # Save image request file
            fs = FileSystemStorage()
            fs.save("uploaded/" + str(myfile.name), myfile)
            # Use machine learning to reoder list post
            reoder = ReoderLanmark()
            self.list_index = reoder.get_new_order(str(fs.location) + str("/uploaded/") + str(myfile.name))
        return self.index(request, str("uploaded/") + str(myfile.name))