from django.db import models


# Create your models here.
class CodeAnalysisRequest(models.Model):
    github_url = models.URLField(max_length=255)
    language = models.CharField(max_length=50)
    texts = models.TextField()