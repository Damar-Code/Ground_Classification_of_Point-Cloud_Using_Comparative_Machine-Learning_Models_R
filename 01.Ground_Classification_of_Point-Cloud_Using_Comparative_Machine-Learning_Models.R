##### Read LAS Data
lapply(c("lidR","raster","sf","ggplot2"), require, character.only=T)


setwd("LAS_location/")
las <- readLAS("name.las")
crs(las) <- "EPSG:28992"
#plot(las)
#las

## CLIP by AOI
aoi <- st_read("shp_location/")
plot(st_geometry(aoi))
#st_area(plot) 

las_aoi <- clip_roi(las, aoi)  
plot(las_aoi, size=3)

##### SEPARATE GROUND AND NONGROUND

lasfilterGround <- las_aoi[las_aoi@data$Classification==2L,]
lasfilterUnclass <- las_aoi[las_aoi@data$Classification!=2L,]

######################################
# Ground Metrics Extraction
######################################

metrics <- point_metrics(lasfilterGround, ~list(Zmin = sd(Z-min(Z))), k = 20)
metrics2 <- point_metrics(lasfilterGround, ~list(slope = atan(sd(Z-mean(Z)/sqrt(sd(X-mean(X))^2+sd(Y-mean(Y))^2)))), k = 20)
metrics3 <- point_metrics(lasfilterGround, ~list(index = sd(Z)+sd(ScanAngle)+sd(Intensity)), k = 20)


is.planar <- function(x, y, z, th1 = 10, th2 = 6) {
  xyz <- cbind(x, y, z)
  cov_m <- cov(xyz)
  eigen_m <- eigen(cov_m)$value
  is_planar <- eigen_m[2] > (th1*eigen_m[3]) && (th2*eigen_m[2]) > eigen_m[1]
  eigen1 <- eigen_m[1]
  eigen2 <- eigen_m[2]
  eigen3 <- eigen_m[3]
  sumEigen <- eigen_m[1]+eigen_m[2]+eigen_m[3]
  anisotropy <- (eigen_m[1]-eigen_m[3])/eigen_m[2]
  planarity <- (eigen_m[2]-eigen_m[3])/eigen_m[1]
  linearity <- (eigen_m[1]-eigen_m[2])/eigen_m[3]
  SV <- (eigen_m[3])/(eigen_m[1]+eigen_m[2]+eigen_m[3])
  Sphericity <- (eigen_m[3])/eigen_m[1]
  return(list(planar = is_planar,eigen1=eigen1,eigen2=eigen2,eigen3=eigen3,sumEigen=sumEigen,anisotropy=anisotropy,planarity=planarity,linearity=linearity,SV=SV,Sphericity=Sphericity))
}
lasfilterGround@data

metrics4 <- lidR::point_metrics(lasfilterGround, ~is.planar(X,Y,Z), k = 20)
metrics4$planar2 <- as.integer(ifelse(metrics4$planar=="TRUE",1,0))

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "Zmin")
lasfilterGround$Zmin[metrics$pointID] <- metrics$Zmin
#plot(lasfilterGround, color = "Zmin",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "slope")
lasfilterGround$slope[metrics2$pointID] <- metrics2$slope
#plot(lasfilterGround, color = "slope",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "index")
lasfilterGround$index[metrics3$pointID] <- metrics3$index
#plot(lasfilterGround, color = "index",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "planar")
lasfilterGround$planar[metrics4$pointID] <- metrics4$planar2
#plot(lasfilterGround, color = "planar",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "eigen1")
lasfilterGround$eigen1[metrics4$pointID] <- metrics4$eigen1
#plot(lasfilterGround, color = "eigen1",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "eigen2")
lasfilterGround$eigen2[metrics4$pointID] <- metrics4$eigen2
#plot(lasfilterGround, color = "eigen2",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "eigen3")
lasfilterGround$eigen3[metrics4$pointID] <- metrics4$eigen3
#plot(lasfilterGround, color = "eigen3",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "sumEigen")
lasfilterGround$sumEigen[metrics4$pointID] <- metrics4$sumEigen
#plot(lasfilterGround, color = "sumEigen",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "anisotropy")
lasfilterGround$anisotropy[metrics4$pointID] <- metrics4$anisotropy
#plot(lasfilterGround, color = "anisotropy",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "planarity")
lasfilterGround$planarity[metrics4$pointID] <- metrics4$planarity
#plot(lasfilterGround, color = "planarity",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "linearity")
lasfilterGround$linearity[metrics4$pointID] <- metrics4$linearity
#plot(lasfilterGround, color = "linearity",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "SV")
lasfilterGround$SV[metrics4$pointID] <- metrics4$SV
#plot(lasfilterGround, color = "SV",legend=T)

lasfilterGround <- add_attribute(lasfilterGround, FALSE, "Sphericity")
lasfilterGround$Sphericity[metrics4$pointID] <- metrics4$Sphericity
plot(lasfilterGround, color = "Sphericity",legend=T)

######################################
# Non Ground Metrics Extraction
######################################

metrics <- point_metrics(lasfilterUnclass, ~list(Zmin = sd(Z-min(Z))), k = 20)
metrics2 <- point_metrics(lasfilterUnclass, ~list(slope = atan(sd(Z-mean(Z)/sqrt(sd(X-mean(X))^2+sd(Y-mean(Y))^2)))), k = 20)
metrics3 <- point_metrics(lasfilterUnclass, ~list(index = sd(Z)+sd(ScanAngle)+sd(Intensity)), k = 20)


is.planar <- function(x, y, z, th1 = 10, th2 = 6) {
  xyz <- cbind(x, y, z)
  cov_m <- cov(xyz)
  eigen_m <- eigen(cov_m)$value
  is_planar <- eigen_m[2] > (th1*eigen_m[3]) && (th2*eigen_m[2]) > eigen_m[1]
  eigen1 <- eigen_m[1]
  eigen2 <- eigen_m[2]
  eigen3 <- eigen_m[3]
  sumEigen <- eigen_m[1]+eigen_m[2]+eigen_m[3]
  anisotropy <- (eigen_m[1]-eigen_m[3])/eigen_m[2]
  planarity <- (eigen_m[2]-eigen_m[3])/eigen_m[1]
  linearity <- (eigen_m[1]-eigen_m[2])/eigen_m[3]
  SV <- (eigen_m[3])/(eigen_m[1]+eigen_m[2]+eigen_m[3])
  Sphericity <- (eigen_m[3])/eigen_m[1]
  return(list(planar = is_planar,eigen1=eigen1,eigen2=eigen2,eigen3=eigen3,sumEigen=sumEigen,anisotropy=anisotropy,planarity=planarity,linearity=linearity,SV=SV,Sphericity=Sphericity))
}
lasfilterUnclass@data

metrics4 <- lidR::point_metrics(lasfilterUnclass, ~is.planar(X,Y,Z), k = 20)
metrics4$planar2 <- as.integer(ifelse(metrics4$planar=="TRUE",1,0))

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "Zmin")
lasfilterUnclass$Zmin[metrics$pointID] <- metrics$Zmin
#plot(lasfilterUnclass, color = "Zmin",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "slope")
lasfilterUnclass$slope[metrics2$pointID] <- metrics2$slope
#plot(lasfilterUnclass, color = "slope",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "index")
lasfilterUnclass$index[metrics3$pointID] <- metrics3$index
#plot(lasfilterUnclass, color = "index",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "planar")
lasfilterUnclass$planar[metrics4$pointID] <- metrics4$planar2
#plot(lasfilterUnclass, color = "planar",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "eigen1")
lasfilterUnclass$eigen1[metrics4$pointID] <- metrics4$eigen1
#plot(lasfilterUnclass, color = "eigen1",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "eigen2")
lasfilterUnclass$eigen2[metrics4$pointID] <- metrics4$eigen2
#plot(lasfilterUnclass, color = "eigen2",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "eigen3")
lasfilterUnclass$eigen3[metrics4$pointID] <- metrics4$eigen3
#plot(lasfilterUnclass, color = "eigen3",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "sumEigen")
lasfilterUnclass$sumEigen[metrics4$pointID] <- metrics4$sumEigen
#plot(lasfilterUnclass, color = "sumEigen",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "anisotropy")
lasfilterUnclass$anisotropy[metrics4$pointID] <- metrics4$anisotropy
#plot(lasfilterUnclass, color = "anisotropy",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "planarity")
lasfilterUnclass$planarity[metrics4$pointID] <- metrics4$planarity
#plot(lasfilterUnclass, color = "planarity",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "linearity")
lasfilterUnclass$linearity[metrics4$pointID] <- metrics4$linearity
#plot(lasfilterUnclass, color = "linearity",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "SV")
lasfilterUnclass$SV[metrics4$pointID] <- metrics4$SV
#plot(lasfilterUnclass, color = "SV",legend=T)

lasfilterUnclass <- add_attribute(lasfilterUnclass, FALSE, "Sphericity")
lasfilterUnclass$Sphericity[metrics4$pointID] <- metrics4$Sphericity
#plot(lasfilterUnclass, color = "Sphericity",legend=T)

nrow(lasfilterUnclass)

###### SPLIT DATA

all.class <- rbind(lasfilterUnclass, lasfilterGround)
df.all.class <- as.data.frame(all.class@data)
df.all.class

library(caret)
set.seed(123)
factors.list <- c("Intensity","ScanAngle","R","G","B","NIR","Zmin","slope","index",            
                  "planar","eigen1","eigen2","eigen3","sumEigen","anisotropy","planarity","linearity","SV","Sphericity")

library(dplyr)
X = dplyr::select(df.all.class, c(factors.list))
y =  dplyr::select(df.all.class, Classification)

dataset <- cbind(X,y)
part.index <- createDataPartition(dataset$Classification, 
                                  p = 0.70,                         
                                  list = FALSE)
X_train <- X[part.index, ]
head(X_train)
X_test <- X[-part.index, ]

y_train <- y[part.index]
head(y_train)
y_test <- y[-part.index]
unique(y_test)

###### Preprocess
process <- preProcess(X_train, method=c("range"))
X_train <- predict(process, X_train)

process <- preProcess(X_test, method=c("range"))
X_test <- predict(process, X_test)


###### FEATURE SELECTION
#### Correlation Matrics
library(corrplot)

#############################
# Feature Selection
#############################
# calculate correlation matrix
cor.df <- X_train
length(cor.df)

correlationMatrix <- cor(cor.df)
corrplot(correlationMatrix, method="number")

## correlation Plot Visualization
col1 <- colorRampPalette(brewer.pal(9, "BrBG"))
corrplot(correlationMatrix, method = "square", addCoef.col = "red", number.cex = 0.6, tl.cex = 1, tl.col = "black", col = col1(100))

# summarize the correlation matrix
print(correlationMatrix)

# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.7)
highlyCorrelated

#remove highly correlated feature
X_train.selected <- dplyr::select(X_train, -(highlyCorrelated))

## add y_train
X_train.selected$Classification <- y_train
X_train.selected
nrow(X_train.selected)

## RFE
set.seed(100)
options(warn=-1)

subsets <- c(1:10, 10, 15, 18)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

df.rfe.ground <- X_train.selected[X_train.selected$Classification == 2,][1:4000,]
df.rfe.non <- X_train.selected[X_train.selected$Classification == 1,][1:4000,]
df.rfe <- rbind(df.rfe.ground, df.rfe.non)

lmProfile <- rfe(x=dplyr::select(df.rfe, -"Classification"), y=df.rfe$Classification,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile
#### plot
varImp(lmProfile)


varimp_data <- data.frame(feature = row.names(varImp(lmProfile)),
                          importance = varImp(lmProfile),
                          row.names = c(1:6))
varimp_data
library(RColorBrewer)
ggplot(data = varimp_data, 
       aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
  geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
  scale_fill_brewer(palette = "Set3") +
  geom_text(aes(label = round(importance, 2)), vjust=1.6, color="black", size=4, fontface = "bold") + 
  theme_bw() + theme(legend.position = "none",
                     axis.text.x=element_text(colour="black", size = 10),
                     axis.text.y=element_text(colour="black", size = 10),
                     text = element_text(size=20))



#####################
## Filtered Factors
#####################
selected.features <- c(row.names(varImp(lmProfile)))

trainDataset <- cbind(dplyr::select(X_train, selected.features), y_train)
colnames(trainDataset)[names(trainDataset) == "y_train"] <- "Classification"
nrow(trainDataset)

###### Modelling
library(caret)
library(caretEnsemble)


## subset the dataset
original.dataset <- trainDataset
ground <- trainDataset[trainDataset$Classification == 2,][1:10000,]
nonground <- trainDataset[trainDataset$Classification == 1,][1:10000,]
trainDataset <- rbind(ground, nonground)
nrow(trainDataset)
trainDataset$Classification

## Target variable as a factor
trainDataset$Classification <- ifelse(trainDataset$Classification == 2, "ground", "nonground")
trainDataset$Classification <- as.factor(trainDataset$Classification)
str(trainDataset$Classification)

seed <- 7
metric <- "Accuracy"
# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE,
                        index=createFolds(trainDataset$Classification, 10))
algorithmList <- c('knn', 'svmRadial',"C5.0","gbm")
set.seed(seed)
models <- caretList(Classification~., data=trainDataset, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)
results$models


## Ensable Model
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)

stack.rf <- caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.rf)


####### SAVE THE MODELS
output_model <- "location/" 

saveRDS(models$knn, paste0(output_model, "KNN.rds"))
saveRDS(models$svmRadial, paste0(output_model, "SVM.rds"))
saveRDS(models$C5.0, paste0(output_model, "C50.rds"))
saveRDS(models$gbm, paste0(output_model, "GBM.rds"))
saveRDS(stack.rf, paste0(output_model, "EnsambleModel.rds"))

### Model Prediction 
X_test.selected <- dplyr::select(X_test, c("Sphericity", "Zmin", "linearity", "Intensity", "planarity","eigen1","index"))


knn <- predict(models$knn, X_test.selected)
svm <- predict(models$svmRadial, X_test.selected)
c50 <- predict(models$C5.0, X_test.selected)
gbm <- predict(models$gbm, X_test.selected)
Ensamble <- predict(stack.rf, X_test.selected)



### Model Validation
y_test <- ifelse(y_test == 2, "ground", "nonground")
y_test <- as.factor(y_test)
str(y_test)

#Creating confusion matrix
CM.Ensamble <- confusionMatrix(data=Ensamble, reference = y_test)
CM.svm <- confusionMatrix(data=svm, reference = y_test)
CM.knn <- confusionMatrix(data=knn, reference = y_test)
CM.c50 <- confusionMatrix(data=c50, reference = y_test)
CM.gbm <- confusionMatrix(data=gbm, reference = y_test)

CM.Ensamble$table
CM.svm$table
CM.c50$table
CM.gbm$table
CM.knn$table

####
## ROC CURVE
library(pROC)

## as numeric
knn.ROC <- ifelse(knn == "ground",2,1)
svm.ROC <- ifelse(svm == "ground",2,1)
c50.ROC <- ifelse(c50 == "ground",2,1)
gbm.ROC <- ifelse(gbm == "ground",2,1)
Ensamble.ROC <- ifelse(Ensamble == "ground",2,1)

ROC_test <- ifelse(y_test == "ground",2,1)
unique(ROC_test)
unique(gbm.ROC)
#define object to plot and calculate AUC
# RF
knn_roc <- roc(ROC_test, knn.ROC)
knn_auc <- round(auc(ROC_test, knn.ROC),5)
knn_auc
# SVM
svm_roc <- roc(ROC_test, svm.ROC)
svm_auc <- round(auc(ROC_test, svm.ROC),5)
svm_auc
# XGBoost
c50_roc <- roc(ROC_test, c50.ROC)
c50_auc <- round(auc(ROC_test, c50.ROC),5)
c50_auc
# Boosted Logistic Regression
gbm_roc <- roc(ROC_test, gbm.ROC)
gbm_auc <- round(auc(ROC_test, gbm.ROC),5)
gbm_auc

# Ensamble
Ensamble_roc <- roc(ROC_test, Ensamble.ROC)
Ensamble_auc <- round(auc(ROC_test, Ensamble.ROC),5)
Ensamble_auc

#create ROC plot
ggroc(list(KNN = knn_roc, SVM = svm_roc, c.50 = c50_roc, GBM = gbm_roc, Ensamble = Ensamble_roc)
      ,size = .8) +
  ggtitle("ROC: Accuracy of Models Prediction") +
  theme_minimal() 

