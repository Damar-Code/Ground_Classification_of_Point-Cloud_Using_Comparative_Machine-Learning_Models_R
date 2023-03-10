# Ground_Classification_of_Point-Cloud_Using_Comparative_Machine-Learning_Models

Ground Classification is fundamental step for impelementing LiDAR in Forest Inventory Purposes to understand the tree structural properties. In "lidR" package it self already provided model for classify ground point, such as: Progressive Morphological Filter (PMF), Cloth Simulation Function (CSF) and Multiscale Curvature Classification (MCC). However, all those models are still not accurate anough for landscape scale, especially for areas with complex terrain. On another hand, those algorithms take an extremely long time of processing for landscape scale. Therefore, be needed Machine Learning for building highly accurate ground classification models, but did not take much processing time.

## METHODE

![Flow Chart_Ground Classification](https://user-images.githubusercontent.com/60123331/211615140-18bb2de3-67f4-4409-9509-a17b6647db22.png)

Figure 1. Workflow of Building Ground Classification Model Using Machine Learning

Builing Ground Classification model in this study are using 19 predictors. There are 6 predictors originally from LiDAR it self, such as intensity, R, G, B, NIR, and scan angle. Moreover, to understand the terrain complexity there are additional geometrical features as the predictors, such as Zmin, slope, index, planar, 1st eigenvalue, 2nd eigenvalue, 3rd eigenvalue, sum Eigen, anisotropy, planarity, linearity, SV, and Sphericity.

Geometric properties are useful for explaining the local geometry of points. These geometric features are nowadays widely applied in LiDAR data processing. It is aimed to improve the accuracy values by extracting these geometric features in multiple scales rather than on a single scale. Geometric features are calculated by the eigenvalues (λ1, λ2, λ3) of the eigenvectors (v1, v2, v3) derived from the covariance matrix of any point p of the point cloud:


![covariance metrics - Copy](https://user-images.githubusercontent.com/60123331/211853261-115dcd72-6ce8-479a-8077-7c782bb67cd8.PNG)


![assf](https://user-images.githubusercontent.com/60123331/211852844-e1ebc282-d8b5-4e98-982d-c29481d770dd.PNG)


In order to obtain robust models, there is a comparison of five models, which is C5.0, GBM, SVM, KNN, and Ensemble. There are around 200,000 point cloud as a dataset, which is 70% for training and 30% for testing. Meanwhile, in the validation phase, there are two steps: first, validation using a testing dataset and second, revalidating the models in landscape scale using actual ground dataset.


Area of Intesest as a training dataset to build machine learning models in this study showed in this picture below. The terrain condition is quite hilly with dense forest canopy. 

![AoI](https://user-images.githubusercontent.com/60123331/211566665-75e690dc-13cf-4871-ae6f-11b3aaeb7f7e.png)

Figure 2. Area of Interest as a Training Dataset

### Feature Selection
Feature selection in sthis study using two steps, Correlation Metrics and Recursive Feature Elimination. Correlation Metrics is usefull to avoid the multicollinearity from the predictors. Using cut-off 0.7 eliminated index, planar, SV, 2nd eigenvalue, 3rd eigenvalue, and sum eigenvalue.

![Corrpot - Resize](https://user-images.githubusercontent.com/60123331/211583983-48c0f339-4d69-4ec8-a58e-a5c21b3ec6d2.png)

Figure 3. Correlation Metrics

Following the removal of predictors with high correlation,  the rest of the predictors are pitted to find the most significant predictors. Figure 4 below shows there are six important variables to classify ground point cloud, there are sphericity, 1st eigenvalue, intensity, linearity, planarity, and index.

![Variable Importance - resize](https://user-images.githubusercontent.com/60123331/211583792-cb679b55-becb-4fa5-8971-2f16f897e3f2.png)

Figure 4. Variable Importance

## RESULT AND DISCUSSION

![model_comparison - resize](https://user-images.githubusercontent.com/60123331/211583738-396298ab-22b7-4fe9-85da-363450b5173c.png)

Figure 5. Accuracy and Kappa

Clearly see that how the algorithms performed in terms of accuracy and kappa. The C5.0 model appears to be the be best performing model overall because of the high accuracy and Kappa. Either way, there are second validation to make final decision on which model will have best peformance.

## Second Validation

Second validation aims to assest the model prediction peformance on the landscape scale. Cross section below shows how the model classified the ground and nonground point cloud. Mostly the model are still having missed classification. Some point cloud that should be classified as a ground point cloud are classified to be an nonground point cloud and the oposite way as well. The Root cause it might derive by two possibility: 1) because the trainining dataset of ground and nonground point cloud is not in a equel number, and 2) The terrain compplexity of the training dataset still not enough. In ths study the training dataset are loceted on dense forest area it lead of lacking of ground point cloud because most of the point cloud are indered by the canopies.

![cross section map](https://user-images.githubusercontent.com/60123331/211581039-c105c088-d344-4932-a4ea-5d374087222f.png)

Figure 6. Cross Section Map

MAPE value from each model are got from its differences value between the actual ground data. It means the differences between elevation value. Sampling point distribution for MAPE analysis are using purposive sampling technique, focused on abnormality area with some outlier are there. The outliers it self are clearly visible from the Digital Elevation Model Map below

![Sampling MAPE - Copy](https://user-images.githubusercontent.com/60123331/211584330-85584766-20e6-4519-bf7c-6542a35fc51b.png)

Figure 7. Sampling Point Distribution for MAPE Assesment

Each models performance in lendscape scale are represents from this figure below. The overall accuracy of the model is quite good basen on the MAPE value only SVM model have a poor prediction. Red box in the picture emphasize the example area with outlier or miss classification. The Ensemble have lower peformance than C5.0 because the impact of the SVM model.


![comparison](https://user-images.githubusercontent.com/60123331/211609241-108e1a09-03a0-4135-9691-6cbf43574c33.png)
![MAPE](https://user-images.githubusercontent.com/60123331/211611038-a7ab234e-dffc-4b08-bbca-967d906473a1.png)

Figure 8. Models Peformance for Landscape Scale

## CONCLUSION
Machine learnining are well proven for making ground classification models. Models quality improvement can be done by two ways: 1) increasing the number of training dataset in a complex terrain, and 2) take out the poor model peformance before make an ensamble model. Moreover, try to implement the another classifier model of machine learning.

## REFERENCES
- Atik, Muhammed Enes, Zaide Duran, and Dursun Zafer Seker. 2021. “Machine Learning-Based Supervised Classification of Point Clouds Using Multiscale Geometric Features.” ISPRS International Journal of Geo-Information. [doi: 10.3390/ijgi10030187](https://www.mendeley.com/catalogue/4476ecc9-d5f0-3af0-8eef-f4ca8a061bb2/?utm_source=desktop&utm_medium=1.19.4&utm_campaign=open_catalog&userDocumentId=%7B64628a51-8ea3-4821-89bd-b85d9e5e10c1%7D)
- Atik ME, Duran Z. An Efficient Ensemble Deep Learning Approach for Semantic Point Cloud Segmentation Based on 3D Geometric Features and Range Images. Sensors (Basel). 2022 Aug 18;22(16):6210. [doi: 10.3390/s22166210](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9416655/)

