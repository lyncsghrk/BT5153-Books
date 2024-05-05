---
title: Books Recommender
emoji: 📚
colorFrom: green
colorTo: red
sdk: static
pinned: false
---

# To see the files, please go to [this huggingface repo](https://huggingface.co/spaces/bt5153-books/README/tree/main)!
As Github LFS only supports up to 2GB of data, we have hosted our full repo on huggingface.

Link to files: [https://huggingface.co/spaces/bt5153-books/README/tree/main](https://huggingface.co/spaces/bt5153-books/README/tree/main)

# Books Recommendation Project (BT5153)

Hello, and welcome to our books recommendation project for BT5153!

# Project Directory

## Source Code
Model codes are stored under `./Books` as `.ipynb` files, and named according to the order they should be run. 
User interface codes for Flask are stored in the root `./` directory, and the html files can be found under `./templates`.

## Data
All data used for the project is stored in `./Data` in the [Huggingface repository](https://huggingface.co/spaces/bt5153-books/README/tree/main).

***WARNING: This huggingface repository is over 18GB, ensure that you have sufficient space on the disk before cloning***

For our submission, we have created a representative subset of our dataset to be included in the zip submission, and can be found in `data05.zip` in the accompanying files. These sample subset files can also be found in the [Github repository](https://github.com/lyncsghrk/BT5153-Books), under the directory `./Data/data05.zip`.

# To run our project in Windows:

## Create a virtual environment (optional) 
Run these commands:
1. `python -m venv venv`
2. `venv\Scripts\activate`
3. `python -m pip install -r requirements.txt`

## Locate python notebooks
All python notebooks can be found in the subdirectory `./Books/`.

## Data preprocessing
***Run all cells in the file `1_data_split.ipynb`.***

## Generating recommendations
***Run all cells in the following files:***
* `2.1_users_similarity.ipynb`
* `2.2_reviews_LDA.ipynb`
* `2.3_description_s2v.ipynb`
* `2.4_genres_w2v.ipynb`
* `2.5_titles_bge_faiss.ipynb`
* `2.6_book_clustering.ipynb`

Do note that some notebooks may take up to a few hours to complete.
The recommendations have been saved and stored under the directory `./Data/Books/Recommend Storage`, in numpy arrays.

***Then, run all cells in the following file to generate recommendation for users:***
`3_book_to_user_converer.ipynb`

## Ensemble model
Run all cells in this file: `4_ensemble_final.ipynb`


# Front-end UI
## Book Recommendation Ensemble Model Interface

This interface generates recommendations, but only for a list of randomly sampled test users from our dataset.

This interface was created on Python version 3.11.4, with requirements listed in `requirements.txt`.
There may be some requirements missed, please install as needed.

All sub-models and the final ensemble classifier model were trained in advance. They are included inside the Data folder.

All data used for live recommendation is in the Data folder. Since the Data folder is too large to be submitted, we will submit a representative subset of the data.

## To start the UI:
**NOTE: Please only run this with the full dataset from [this git repository](https://huggingface.co/spaces/bt5153-books/README/tree/main)!!** If not, an error will occur and there will not be any results...

Start the interface with `python -m flask run`.

If for some reason app does not start, try running `python app.py`.

Server should be running on `127.0.0.1:5000`


----------------------------------------------------------------

# Project Abstract
This report presents an enhanced book recommendation system to improve recommendation precision. By integrating unstructured text data and diverse data sources, the proposed system offers more robust recommendations tailored to individual users, ultimately improving user retention rates.

Utilizing a Goodreads dataset from 2017, comprising book information, user-tagged genres, and user reviews, we built and trained six distinct models based on different data types. These models were then combined into an ensemble logistic regression model, outperforming individual models in precision and exhibiting higher F1 scores in binary classification for book recommendations.

While the ensemble model requires more computational resources than the user similarity model, it effectively mitigates popularity bias, a common issue in naive recommendation systems. Finally, the system's user interface, developed with Flask, offers transparent recommendations with explainability graphs, enhancing user trust and experience. 

Overall, the enhanced book recommendation system shows promising results and has the potential to outperform the naive user similarity model with further data refinement and model training.

*For full project description, see the report file in submission.*

## Members:
* Ang Kai En (A0221945E)
* Meritxell Camp Garcia (A0280366B)
* Sidharth Pahuja (A0218880X)
* Sim Jun You (A0200198L)
* Sim Yew Chong (A0189487A)
