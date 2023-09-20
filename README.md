##🍀 Plant Disease Detection 🍃🍂
Plant Disease is necessary for every farmer so we are created Plant disease detection using Deep learning. In which we are using convolutional Neural Network for classifying Leaf images into 39 Different Categories. The Convolutional Neural Code build in Pytorch Framework. For Training we are using Plant village dataset.

### About Dataset 🍁
- Dataset link -> https://data.mendeley.com/datasets/tywbtsjrjv/1
  #### Description
  In this data-set, 39 different classes of plant leaf and background images are available.  The data-set containing 61,486 images. We used six different augmentation techniques for increasing the data-set size. The techniques are image flipping, Gamma correction, noise injection, PCA color augmentation, rotation, and Scaling.
    
#### Dataset contains 39 classes:
|                         |                         |                         |
|-------------------------|-------------------------|-------------------------|
| Apple_scab              | Grape_black_measles     | Squash_powdery_mildew   |
| Apple_black_rot         | Grape_leaf_blight       | Strawberry_healthy      |
| Apple_cedar_apple_rust  | Grape_healthy           | Strawberry_leaf_scorch  |
| Apple_healthy           | Orange_haunglongbing    | Tomato_bacterial_spot   |
| Background_without_leaves | Peach_bacterial_spot  | Tomato_early_blight     |
| Blueberry_healthy       | Pepper_bacterial_spot   | Tomato_healthy          |
| Cherry_powdery_mildew   | Pepper_healthy          | Tomato_late_blight      |
| Cherry_healthy          | Potato_early_blight    | Tomato_leaf_mold        |
| Corn_gray_leaf_spot     | Potato_healthy          | Tomato_septoria_leaf_spot |
| Corn_common_rust        | Potato_late_blight     | Tomato_spider_mites_two-spotted_spider_mite |
| Corn_northern_leaf_blight | Raspberry_healthy    | Tomato_target_spot      |
| Corn_healthy            | Soybean_healthy         | Tomato_mosaic_virus     |
| Grape_black_rot         |                        | Tomato_yellow_leaf_curl_virus |

    
#### Categories
Fungal Diseases in Plant, Bacterial Diseases in Plant, Plant Diseases, Deep Learning


###▶️Run Project in your Machine
* You must have python install in your machine.
* Create and activate a Python Virtual Environment
* Install all the dependencies using below command
    `pip install -r requirements.txt`
* Run the Flask app `python app.py`
* You can also use downloaded file in `Model` Section and play with it using Jupyter Notebook.

###🧪Testing Images

* If you do not have leaf images then you can use test images located in test_images folder
* Each Image have it's disease name so you can verify model is working perfact or not.


