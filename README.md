# TopoBD
Code for the paper Topological Deep Learning for Enhanced Diagnosis of Blood Disorders <br />

Our project explores the application of topological methods to improve biomedical image classification of rare blood disorders. We utilize topological data analysis in the form of Betti vectors to create more robust and accurate models to classify these disorders. <br /> 

## Table of Contents






## Installation
To run the models, please download the datasets we used from [Zenodo](https://zenodo.org/records/14474907?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjBhYWMwYzY0LTQyZGEtNDQwMy04MTk2LTJjOTI4YzMyN2QzYSIsImRhdGEiOnt9LCJyYW5kb20iOiI4MTNiM2Q3Y2RmZDdlMjcxZTJlMDA0ODY1ZjhhN2ZmMCJ9.zYRSnDcx8W8UIAioiWtr6n4kBi0_hbFoTCf51sV0ENvBe9DMDr5TpVELOgnwpj0tnOVdnMZe-DFT6heFxOk18A). To access the ALL-IDB2 dataset please request access from the creators [here](https://scotti.di.unimi.it/all/). Each .npz file contains an 'images' array containing the pixel values of each image and 'labels' array containing the class of each image (0 for normal, 1 for abnormal). The AML .npz contains an extra array 'binary_labels' to classify normal images from abormal for binary classification. Each .csv contains the 400-dimensional betti vector for each image. After downloading the files, please download the required dependencies as mentioned in the requirements.txt. 

## Usage



## Data
The datasets used for this project are not included in the repository. However, the links to the datasets can be found below: <br />
* [ALL](https://scotti.di.unimi.it/all/#)
* [AML](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/)
* [Malaria](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)
* [NIAID](https://data.niaid.nih.gov/resources?id=mendeley_38jtn4nzs6)
      
## Results and Figures

The specific results of our experiments can be found in our paper. To summarize, topological features enhanced the performance of baseline models in all the datasets we used.

### Accuracy Comparison of Different Models
![RareBloodDiseaseAccuracyPlot](https://github.com/user-attachments/assets/c857c54d-b20a-4faa-8444-8632f895b601)


### AUC Comparison of Different Models 
![RareBloodDiseaseAUCPlot](https://github.com/user-attachments/assets/dab48593-0278-458d-8072-54e8ca5767e9)


## Acknowledgements 
* We would like to thank the authors of the [datasets](#data) for their time and effort in developing these valuable resources
