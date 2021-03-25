
from django.contrib import admin
from django.urls import path
from face_detector import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('face_detector/detect/',views.detect)
]
