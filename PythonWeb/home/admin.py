from django.contrib import admin

from .models import Lanmark
# Register your models here.
class LanmarkAdmin(admin.ModelAdmin):
    list_display = ["id", "title", "body", "date", "image", "num"]
admin.site.register(Lanmark, LanmarkAdmin)