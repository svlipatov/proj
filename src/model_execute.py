import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv

def preprocess_images(images):
    """
    Preprocess image for the model.
    """
    preprocess = transforms.Compose([
        transforms.Resize([70, 70]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images_tensor = [preprocess(image) for image in images]
    image_batch = torch.stack(images_tensor)

    return image_batch

def output_to_names(output):
    """
    Converts model outputs to category names names.
    """
    with open('model/cat.csv') as file:
        reader = csv.reader(file)
        cat_list = list(reader)[0]

    names = []
    for prediction in output:
        probabilities = torch.nn.functional.softmax(prediction, dim=0)
        index = probabilities.argmax()
        names.append(cat_list[index])
    return names

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


if __name__ == "__main__":
    pkl_filename = "model/pickle_model.pkl"
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

    tensor = torch.tensor([[-1.8637, -1.6411, -1.5038, -2.9645, -1.8477, 6.5004], [-1.6067, -1.6597, -1.0925, 5.1295, -1.6491, -1.4739], [-0.2427, -0.6140, -1.1936, -2.1147, 4.8429, -2.0129]])
    print(output_to_names(tensor))
