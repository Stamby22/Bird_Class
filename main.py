from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
#import uvicorn
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
#from keras.saving import load_model
#from keras.utils import img_to_array
from numpy import expand_dims, argmax
from json import load
import os

app = FastAPI()
IMG_SIZE = 300

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading the classes from the JSON file
current_working_directory = os.getcwd()
bird_classes_filepath = os.path.join(current_working_directory,"Models","bird_classes.json")
with open(bird_classes_filepath, 'r') as json_file:
    bird_classes = load(json_file)
# create invert dictionary
bird_species = {v: k for k, v in bird_classes.items()}

# load trained model
best_model_file_name = os.path.join(current_working_directory,"Models","Bird_EffiB3_04_FT.h5")
MODEL = load_model(best_model_file_name)           

# get probabilities of the best predictions
def get_probabilities(prob_list, trashold):
    #prob_list = model_labels[ind]
    prob_dict = {}
    i = 0
    for prob in prob_list:
        prob_dict[round(100*prob)] = str(bird_species.get(i)).replace('_', ' ')
        i += 1
    # myKeys = list(prob_dict.keys())
    myKeys = list(num for num in prob_dict.keys() if num > trashold)
    myKeys.sort(reverse = True)
    # myKeys = myKeys[:num_best]  # [cislo for cislo in cisla if cislo > 30]
    sorted_list = ['\n'+str(i)+'% '+ prob_dict[i] for i in myKeys]
    return sorted_list


@app.get("/ping")
async def ping():
    return "hello , i am alive"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_resized = img_to_array(img_resized)
        img_batch = expand_dims(img_resized, axis=0)
        predictions = MODEL.predict(img_batch, verbose=0)
        prob_list = get_probabilities(predictions[0], 1)
        predictions = argmax(predictions, axis=1)
        print(predictions[0])
        predicted_class = bird_species.get(int(predictions[0]))
        print(predicted_class)

        return {
            'class': str(predicted_class).replace('_', ' '),
            'confidence': prob_list
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("main.html", "r") as file:
        return file.read()
