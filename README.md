---
title: Books Recommender
emoji: 📚
colorFrom: green
colorTo: red
sdk: static
pinned: false
---

# To see the files, please go to the Files tab!
Link to files: [https://huggingface.co/spaces/bt5153-books/README/tree/main](https://huggingface.co/spaces/bt5153-books/README/tree/main)

# Books Recommendation Project (BT5153)

Hello, and welcome to our books recommendation project for BT5153!

# Project Directory
## Front-end UI
### Book Recommendation Ensemble Model Interface

This interface generates recommendations, but only for a list of randomly sampled test users from our dataset.

This interface was created on Python version 3.11.4, with requirements listed in `requirements.txt`.
There may be some requirements missed, please install as needed.

All sub-models and the final ensemble classifier model were trained in advance. They are included inside the Data folder.

All data used for live recommendation is in the Data folder. Since the Data folder is too large to be submitted, we will submit a representative subset of the data.

### To start the UI:
**NOTE: Please only run this with the full dataset from [this git repository](https://huggingface.co/spaces/bt5153-books/README/tree/main)!!** If not, there will not be any results...

Start the interface with `python -m flask run`.

If for some reason app does not start, try running `python app.py`.

Server should be running on `127.0.0.1:5000`

## Source Code
Codes are stored under `./Books` as `.ipynb` files, and named according to the order they should be run. 

## Data
Data used for the project is stored in `./Data`. 

Raw data, retrieved from the Goodreads dataset [here](https://mengtingwan.github.io/data/goodreads.html), can be found under `./raw-data`. 

For our submission, we have created a representative subset of our dataset to be included in the zip submission, and can be found in `./Data-sub`.

# To run our project in Windows:

## Create a virtual environment (optional) 
Run these commands:
1. `python -m venv venv`
2. `venv\Scripts\activate`
3. `python -m pip install -r requirements.txt`

## Locate python notebooks
All python notebooks can be found in the subdirectory `./Books/`.

## Data preprocessing
Run all cells in the file `1_data_split.ipynb`.

## Generating recommendations
Run all cells in the following files:
* `2.1_users_similarity.ipynb`
* `2.2_reviews_LDA.ipynb`
* `2.3_description_s2v.ipynb`
* `2.4_genres_w2v.ipynb`
* `2.5_titles_bge_faiss.ipynb`
* `2.6_book_clustering.ipynb`

Then, run the following file to generate recommendation for users:
`3_book_to_user_converer.ipynb`

## Ensemble model
Run this file: `4_ensemble_final.ipynb`

----------------------------------------------------------------

# Project Description
In response to the overwhelming number of book choices online, which often leads to decision paralysis and wasted time, we propose the implementation of a Natural Language Processing (NLP) powered recommendation system to address this challenge.

For full project description, see the report file in submission.

## Members:
* Ang Kai En (A0221945E)
* Meritxell Camp Garcia (A0280366B)
* Sidharth Pahuja (A0218880X)
* Sim Jun You (A0200198L)
* Sim Yew Chong (A0189487A)
