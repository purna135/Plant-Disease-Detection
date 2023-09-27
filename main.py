import os
from flask import Flask, redirect, render_template, request
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from models.model import ResNet9, get_default_device, to_device
import torch


plants = pd.read_csv('./data/plants_name_img.csv',encoding='cp1252')
disease_info = pd.read_csv('./data/disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('./data/supplement_info.csv' , encoding='cp1252')

device = get_default_device()
plant_classes = 38
saved_model = torch.load("./models/plant-disease-model.pth", map_location=device)
model = ResNet9(3, plant_classes)
model.load_state_dict(saved_model)
model.eval()
model = to_device(model, device)

def prediction(image_path):
    image = Image.open(image_path)

    # Resize the image to the same size as the training images
    transform = transforms.Resize((256, 256))
    resized_image = transform(image)

    # Convert the image to a tensor
    transform = transforms.ToTensor()
    image_tensor = transform(resized_image)
    
    # Convert to a batch of 1
    xb = to_device(image_tensor.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label

    return preds[0].item()


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html', plants_name = list(plants['Plant_Name']), plants_img_url = list(plants['Image_url']))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect("/")
    else:
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        pred = prediction(file_path)
        
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        #delete image
        os.remove(file_path)
        
        return render_template('predict.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/code')
def code():
    return render_template('notebook.html')
    
if __name__ == '__main__':
    app.run(debug=True)
