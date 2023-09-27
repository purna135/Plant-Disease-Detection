# ☘️ PLANT DISEASE CLASSIFICATION USING RESNET-9 ☘️ 🍃🍂
Plant Disease is necessary for every farmer so we are created Plant disease detection using Deep learning. In which we are using convolutional Neural Network for classifying Leaf images into 38 Different Categories. The Convolutional Neural Code build in Pytorch Framework. For Training we are using Plant village dataset.

## 📝About dataset 🍁
- Dataset link -> https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
  ### Description
  This dataset is created using offline augmentation from the original dataset. The original PlantVillage Dataset can be found [here](https://github.com/spMohanty/PlantVillage-Dataset).This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.

## 🎯Goal of the Project
Goal is clear and simple. We need to build a model, which can classify between healthy and diseased crop leaves and also if the crop have any disease, predict which disease is it.

## ▶️Run Project in your Machine
* You must have python install in your machine.
* Create and activate a Python Virtual Environment
* Install all the dependencies using below command
    `pip install -r requirements.txt`
* Run the Flask app `python app.py`
* You can also use downloaded notebook in `models` folder and play with it using Jupyter Notebook.

## 🧪Testing Images

* If you do not have leaf images then you can use test images located in test_images folder
* Each Image have it's disease name so you can verify model is working perfact or not.


## 📒Kaggle Notebook
Learn more details from this kaggle notebook: [Plant Disease Detection - ResNet](https://www.kaggle.com/code/purna135/plant-disease-detection-resnet?kernelSessionId=144440071)