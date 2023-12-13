# nfl-data-bowl
This is the repo for my 2024 NFL Data Bowl Submission. 

__NFL Big Data Bowl 2024 Submission:__ University Track <br>
__Author:__ Jack Friedman, Dartmouth College <br>
__Date Submitted:__ December 13, 2023 <br>
__Project Submission:__ <br>
__Project Overview:__ The goal of this project is to predict tackle location on NFL rushing plays given data until the time of handoff. <br>

## Repository overview

__assets:__ Contains figures and tables used in the paper <br>
__data:__ Contains all data used in the project, including 2020 and 2024 NFL Big Data Bowl tracking data <br>
__data-exploration:__ Contains exploratory data analysis code used to generate the distribution of tackle locations/frame count graphs for the paper <br>
__multi-modal:__ Contains code to train the multi-modal model and the best-performing model state dictionary 
__preprocessing:__ Contains data loading and preprocessing modules
__tabnet-model:__ Contains code to train the TabNet model from the play-level data and the model's state dictionary 
__vit-model:__ Contains code to train the ViT model, the ViT state dictionary pretrained on 2020 data, and the model's state dictionary trained on 2024 data 
__vivit-model:__ Contains code to the ViViT model and the model's state dictionary 
