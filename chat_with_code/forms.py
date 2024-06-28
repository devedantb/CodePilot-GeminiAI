from django.forms import ModelForm
from .models import CodeAnalysisRequest

class Form_CodeAnalysisRequest(ModelForm):
    class Meta:
        model = CodeAnalysisRequest
        fields = "__all__"