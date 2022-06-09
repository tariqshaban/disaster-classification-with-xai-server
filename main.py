import base64
import io
import os
import cv2
import numpy as np
import requests
import tensorflow as tf
import gc

from random import randint
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from keras import Sequential
from keras.applications import ResNet50
from keras.engine.base_layer import Layer
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from lime import lime_image
from skimage.segmentation import mark_boundaries
from starlette.middleware.cors import CORSMiddleware

from assets.gradcam import grad_cam, grad_cam_plus
from assets.utils import preprocess_image

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def explain_image_lime(img, model):
    preprocessed_image = tf.keras.applications.resnet50.preprocess_input(img)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.asanyarray(preprocessed_image).astype('double'), model.predict,
                                             top_labels=5, hide_color=0, num_samples=10)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=1000,
                                                hide_rest=False)

    return mark_boundaries(img, mask)


def __show_img_with_heat(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap * 255).astype('uint8')
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img


def explain_image_grad_cam(img, model):
    image_path = f'./assets/buffer{randint(0, 99999)}.jpg'

    Image.fromarray(img).save(image_path)

    img = preprocess_image(image_path, target_size=(224, 224))

    heatmap = grad_cam(
        model, img,
        layer_name=model.layers[1].name,
    )

    img_arr = __show_img_with_heat(image_path, heatmap)

    os.remove(image_path)

    return img_arr


def explain_image_grad_cam_plus_plus(img, model):
    image_path = './assets/buffer.jpg'

    Image.fromarray(img).save(image_path)

    img = preprocess_image(image_path, target_size=(224, 224))

    heatmap = grad_cam_plus(
        model, img,
        layer_name=model.layers[1].name,
    )

    img_arr = __show_img_with_heat(image_path, heatmap)

    os.remove(image_path)

    return img_arr


def predict_image_class(img, model):
    classes = [
        'infrastructure', 'land_slide', 'non_damage_buildings_street', 'non_damage_wildlife_forest',
        'sea', 'urban_fire', 'water_disaster', 'wild_fire'
    ]

    img = np.expand_dims(img, axis=0)
    tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    return classes[np.argmax(model.predict(tensor))]


def image_array_to_base64(img):
    im_png = Image.fromarray(img, 'RGB')
    bytes_io = io.BytesIO()
    im_png.save(bytes_io, format='PNG')
    bytes_image = bytes_io.getvalue()
    return base64.b64encode(bytes_image)


@app.on_event('startup')
async def startup_event():
    r = requests.get(
        'https://raw.githubusercontent.com/tariqshaban/disaster-classification-with-xai-server/master/assets/model_weights.h5')

    with open('assets/model_weights.h5', 'wb') as f:
        f.write(r.content)

    gc.collect()


@app.get('/classify_image')
async def classify_image(file: UploadFile = File(...)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Layer())
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.load_weights('./assets/model_weights.h5')

    del base_model
    gc.collect()

    img = Image.open(file.file)
    img = img.resize((224, 224))
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])[0]

    lime_image_xai = (explain_image_lime(img, model) * 255).astype('uint8')
    lime_image_xai_response = image_array_to_base64(lime_image_xai)

    grad_cam_image_xai = explain_image_grad_cam(img, model)
    grad_cam_image_xai_response = image_array_to_base64(grad_cam_image_xai)

    grad_cam_plus_plus_image_xai = explain_image_grad_cam_plus_plus(img, model)
    grad_cam_plus_plus_image_xai_response = image_array_to_base64(grad_cam_plus_plus_image_xai)

    classification = predict_image_class(img, model)

    del grad_cam_image_xai
    del grad_cam_plus_plus_image_xai
    del img
    gc.collect()

    return {
        'lime': lime_image_xai_response,
        'grad_cam': grad_cam_image_xai_response,
        'grad_cam_plus_plus': grad_cam_plus_plus_image_xai_response,
        'classification': classification
    }
