import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision

def check_photo(name, photo):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(photo)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    print(name, output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(name, probabilities)

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

model.eval()
# sample execution (requires torchvision)

gates_photo = Image.open("gates500.jpg")
musk_photo = Image.open("mask.jpg")
bezos_photo = Image.open("bezos500.jpg")
zuker_photo = Image.open("zuckerberg500.jpg")
jobs_photo = Image.open("jobs500.jpg")
test_photos_dict = {'gates':gates_photo, 'musk':musk_photo, 'bezos':bezos_photo,'zuker': zuker_photo,'jobs': jobs_photo}
for name in test_photos_dict:
    check_photo(name, test_photos_dict[name])
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(test_photos_list)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
#
# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')
#
# with torch.no_grad():
#     output = model(input_batch)
# # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
# print(output[0])
# print(model)

# print(probabilities)