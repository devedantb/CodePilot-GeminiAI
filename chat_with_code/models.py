from django.db import models
from django.contrib.auth.models import User

# Create your models here.
# class PilotUser(User):
#     name = models.CharField(max_length=50)
#     username = models.CharField(max_length=50, unique=True)
#     email = models.EmailField(unique=True)
#     password = models.CharField(max_length=255)


class CodeAnalysisRequest(models.Model):
    github_url = models.URLField(max_length=255,null=True)
    language = models.CharField(max_length=50,null=True)
    chat_history = models.TextField(null=True)
    texts = models.TextField(null=True)
    