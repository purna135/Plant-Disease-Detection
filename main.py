import os
from flask import Flask, redirect, render_template, request
import numpy as np
import pandas as pd
# from PIL import Image
# import torchvision.transforms.functional as TF
# from model import model
# import torch


plants = pd.read_csv('./data/plants_name_img.csv',encoding='cp1252')
disease_info = pd.read_csv('./data/disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('./data/supplement_info.csv' , encoding='cp1252')

# model = model.Model(39)    
# model.load_state_dict(torch.load("./model/plant_disease_model_1_latest.pt"))
# model.eval()

# def prediction(image_path):
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     input_data = TF.to_tensor(image)
#     input_data = input_data.view((-1, 3, 224, 224))
#     output = model(input_data)
#     output = output.detach().numpy()
#     index = np.argmax(output)
#     return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html', plants_name = list(plants['Plant_Name']), plants_img_url = list(plants['Image_url']))

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'GET':
#         return redirect("/")
#     else:
#         image = request.files['image']
#         filename = image.filename
#         file_path = os.path.join('static/uploads', filename)
#         image.save(file_path)
#         pred = prediction(file_path)
#         title = disease_info['disease_name'][pred]
#         description =disease_info['description'][pred]
#         prevent = disease_info['Possible Steps'][pred]
#         image_url = disease_info['image_url'][pred]
#         supplement_name = supplement_info['supplement name'][pred]
#         supplement_image_url = supplement_info['supplement image'][pred]
#         supplement_buy_link = supplement_info['buy link'][pred]
#         #delete image
#         os.remove(file_path)
#         return render_template('predict.html' , title = title , desc = description , prevent = prevent , 
#                                image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)
        
if __name__ == '__main__':
    app.run(debug=True)
