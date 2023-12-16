import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision

# Запуск модели для распознания фото
def check_photo1(model, categorias, photo):
    # Тот же формат фото, что и при обучении
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
    transforms.Resize([70, 70]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAutocontrast(),
    transforms.RandomEqualize(),
    transforms.ToTensor(),
    normalize
])
    batch = preprocess(photo).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    return categorias[class_id], score


