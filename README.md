# Ground_Classification_of_Point-Cloud_Using_Comparative_Machine-Learning_Models

Ground Classification is fundamental step for impelementing LiDAR in Forest Inventory Purposes to understand the tree structural properties. In "lidR" package it self already provided model for classify ground point, such as: Progressive Morphological Filter (PMF), Cloth Simulation Function (CSF) and Multiscale Curvature Classification (MCC). However, all those models are still not accurate anough for landscape scale, especially for areas with complex terrain. On another hand, those algorithms take an extremely long time of processing for landscape scale. Therefore, be needed Machine Learning for building highly accurate ground classification models, but did not take much processing time.

## Methode
Builing Ground Classification model in this study are using geometrical feature as the predictors, such as: Zmin, slope, index, planar, eigen, sum eigen, anistropy, planarity, linearity, SV, and Sphericity.

In order to obtain robust models, there is a comparison of five models, which is C5.0, GBM, SVM, KNN, and Ensemble. There are around 200,000 point cloud as a dataset, which is 70% for training and 30% for testing. Meanwhile, in the validation phase, there are two steps: first, validation using a testing dataset and second, revalidating the models in landscape scale using actual ground dataset.

![Flow Chart_Ground Classification](https://user-images.githubusercontent.com/60123331/211553713-941a7249-f77a-4dbe-bcf9-40de2c9da345.png)

Figure 1. Workflow of Building Ground Classification Model Using Machine Learning

Figure 2 below shows the Area of Intesest as a training dataset to build machine learning models in this study. Terrain condition is quite hilly the dense forest canopy. In total 

![AoI](https://user-images.githubusercontent.com/60123331/211566665-75e690dc-13cf-4871-ae6f-11b3aaeb7f7e.png)

Figure 2. 
