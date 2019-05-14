from django.db import models

# Create your models here.
class Post(models.Model):
    title = models.CharField(max_length=1000)
    body = models.TextField()
    date = models.DateTimeField(auto_now_add=True)
    image = models.TextField(max_length=1000)
    
    def __str__(self):
        return self.title