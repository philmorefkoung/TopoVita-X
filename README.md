# TopoVitaX
Code for the paper Diagnosis of Blood Diseases and Disorders with Topological Deep Learning <br />

Our project explores the application of topological methods to improve biomedical image classification of rare blood disorders. We utilize topological data analysis in the form of Betti vectors to create more robust and accurate models to classify these disorders. <br /> 

## Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Data](#data)
* [Results](#results)
* [Acknowledgements](#acknowledgements)





## Installation
Please refer to the requirements.txt file to install the dependencies

## Usage
Provided below is an example to load the data: <br />
```
import numpy as np
import pandas as pd

image_train = np.load('babesia-train.npz')['images'] 
image_val = np.load('babesia-val.npz')['images'] 
image_test = np.load('babesia-test.npz')['images'] 

betti_train = pd.read_csv('babesia-betti-train.csv')
betti_val = pd.read_csv('babesia-betti-val.csv')
betti_test = pd.read_csv('babesia-betti-test.csv')

labels_train = np.load('babesia-train.npz')['labels']
labels_val = np.load('babesia-val.npz')['labels']
labels_test = np.load('babesia-test.npz')['labels']
```
Betti vectors should be provided in csv format and 400 dimensional in length <br />
Image data and lables should be stored in an npz with 'images' and 'labels' array 



## Data
Due to file size limitations, we cannot provide the datasets in this repository. However, the links to the original datasets can be found below: <br />
* [ALL](https://scotti.di.unimi.it/all/#)
* [AML](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/)
* [Malaria](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)
* [NIAID](https://data.niaid.nih.gov/resources?id=mendeley_38jtn4nzs6) <br />

Please note that only the ALL-IDB2 dataset requires permissions to access 
## Results

The specific results of our experiments can be found in our paper. To summarize, topological features enhanced the performance of baseline models in all the datasets we used.

### Accuracy Comparison of Different Models
![RareBloodDiseaseAccuracyPlot](https://github.com/user-attachments/assets/c857c54d-b20a-4faa-8444-8632f895b601)


### AUC Comparison of Different Models 
![RareBloodDiseaseAUCPlot](https://github.com/user-attachments/assets/dab48593-0278-458d-8072-54e8ca5767e9)


## Acknowledgements 
* We would like to thank the authors of the [datasets](#data) for their time and effort in developing these valuable resources
