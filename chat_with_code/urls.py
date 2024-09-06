from django.urls import path
from . import views

urlpatterns = [
    path('getrepo/', views.GetRepoData,name='getrepo'),
    path('chat/', views.GenerateResponse,name='chat'),
    path('', views.GenerateResponse,name='chat'),
]