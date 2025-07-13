

# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np
# import random

# app = FastAPI()

# kmeans_model = joblib.load('kmeans_model.pkl')
# scaler = joblib.load('scaler.pkl')

# categories = [
#     'Lake', 'Mountainous', 'Hill Station', 'Waterfall',
#     'Fort', 'Coastal', 'Valley', 'Temple',
#     'Mine', 'Monument', 'Museum', 'Resort',
#     'Desert', 'Cave'
# ]

# cluster_to_category_indices = {
#     0: [0, 1, 2, 3],
#     1: [4, 5, 6, 7],
#     2: [8, 9, 10, 11],
#     3: [12, 13]
# }

# class ClickCounts(BaseModel):
#     click_counts: list

# @app.post("/aiinterest")
# def predict_interest(data: ClickCounts):
#     try:
#         counts_array = np.array([data.click_counts])
#         scaled_counts = scaler.transform(counts_array)
#         cluster = kmeans_model.predict(scaled_counts)[0]

#         # Get the category indices list for the predicted cluster
#         category_indices = cluster_to_category_indices.get(cluster, [])
        
#         if not category_indices:
#             return {"error": "No categories mapped for this cluster"}

#         # Randomly select one category from the group
#         selected_index = random.choice(category_indices)
#         predicted_category = categories[selected_index]

#         return {
#             "cluster": int(cluster),
#             "predicted_category": predicted_category
#         }

#     except Exception as e:
#         return {"error": str(e)}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import random

# Initialize FastAPI
app = FastAPI()

# Allow CORS for testing from anywhere (you can restrict this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained models
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define your categories and cluster mapping
categories = [
    'Lake', 'Mountainous', 'Hill Station', 'Waterfall',
    'Fort', 'Coastal', 'Valley', 'Temple',
    'Mine', 'Monument', 'Museum', 'Resort',
    'Desert', 'Cave'
]

cluster_to_category_indices = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12, 13]
}

# Pydantic input model
class ClickCounts(BaseModel):
    click_counts: list

# Home route for testing
@app.get("/")
def read_root():
    return {"message": "AI Interest Inference API is running"}

# AI interest prediction endpoint
@app.post("/aiinterest")
def predict_interest(data: ClickCounts):
    try:
        counts_array = np.array([data.click_counts])
        scaled_counts = scaler.transform(counts_array)
        cluster = kmeans_model.predict(scaled_counts)[0]

        category_indices = cluster_to_category_indices.get(cluster, [])
        if not category_indices:
            return {"error": "No categories mapped for this cluster"}

        selected_index = random.choice(category_indices)
        predicted_category = categories[selected_index]

        return {
            "cluster": int(cluster),
            "predicted_category": predicted_category
        }

    except Exception as e:
        return {"error": str(e)}
