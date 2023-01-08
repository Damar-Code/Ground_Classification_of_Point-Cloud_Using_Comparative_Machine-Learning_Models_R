###########################################################################################################
## CREATE GEOMETRY MATRICS "Sphericity", "Zmin", "linearity", "Intensity", "planarity","eigen1" and "index"
###########################################################################################################

metrics <- point_metrics(las, ~list(Zmin = sd(Z-min(Z))), k = 20)
metrics2 <- point_metrics(las, ~list(slope = atan(sd(Z-mean(Z)/sqrt(sd(X-mean(X))^2+sd(Y-mean(Y))^2)))), k = 20)
metrics3 <- point_metrics(las, ~list(index = sd(Z)+sd(ScanAngle)+sd(Intensity)), k = 20)


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
las@data

metrics4 <- lidR::point_metrics(las, ~is.planar(X,Y,Z), k = 20)
metrics4$planar2 <- as.integer(ifelse(metrics4$planar=="TRUE",1,0))

las <- add_attribute(las, FALSE, "Zmin")
las$Zmin[metrics$pointID] <- metrics$Zmin
#plot(las, color = "Zmin",legend=T)

las <- add_attribute(las, FALSE, "index")
las$index[metrics3$pointID] <- metrics3$index
#plot(las, color = "index",legend=T)

las <- add_attribute(las, FALSE, "planar")
las$planar[metrics4$pointID] <- metrics4$planar2
#plot(las, color = "planar",legend=T)

las <- add_attribute(las, FALSE, "eigen1")
las$eigen1[metrics4$pointID] <- metrics4$eigen1
#plot(las, color = "eigen1",legend=T)

las <- add_attribute(las, FALSE, "planarity")
las$planarity[metrics4$pointID] <- metrics4$planarity
#plot(las, color = "planarity",legend=T)

las <- add_attribute(las, FALSE, "linearity")
las$linearity[metrics4$pointID] <- metrics4$linearity
#plot(las, color = "linearity",legend=T)

las <- add_attribute(las, FALSE, "Sphericity")
las$Sphericity[metrics4$pointID] <- metrics4$Sphericity
#plot(las, color = "Sphericity",legend=T)

###########################
## MODEL IMPLEMENTATION
###########################

#c.50
c50.classify <- predict(models$C5.0, data.frame(las@data))
c50.classify <- ifelse(c50.classify == "ground",2,1)
c50.classify
#svm
svm.classify <- predict(models$svmRadial, data.frame(las@data))
svm.classify <- ifelse(svm.classify == "ground",2,1)
svm.classify
#gbm
gbm.classify <- predict(models$gbm, data.frame(las@data))
gbm.classify <- ifelse(gbm.classify == "ground",2,1)
gbm.classify
#knn
knn.classify <- predict(models$knn, data.frame(las@data))
knn.classify <- ifelse(knn.classify == "ground",2,1)
knn.classify
#Ensamble
Ensamble.classify <- predict(stack.rf, data.frame(las@data))
Ensamble.classify <- ifelse(Ensamble.classify == "ground",2,1)
Ensamble.classify


las.c50 <- las
las.knn <- las
las.svm <- las
las.gbm <- las
las.Ensamble <- las

las.c50$Classification <- as.integer(c50.classify)
las.svm$Classification <- as.integer(svm.classify)
las.gbm$Classification <- as.integer(gbm.classify)
las.knn$Classification <- as.integer(knn.classify)
las.Ensamble$Classification <- as.integer(Ensamble.classify)

## MBA ALGORITHM FOR BUILD DTM 
mba <- function(n = 1, m = 1, h = 8, extend = TRUE) {
  f <- function(las, where) {
    res <- MBA::mba.points(las@data, where, n, m , h, extend)
    return(res$xyz.est[,3])
  }
  
  f <- plugin_dtm(f)
  return(f)
}

## CREATE DTM
dtm_c50 <- grid_terrain(las.c50, res = 0.2, algorithm =  mba())
dtm_svm <- grid_terrain(las.svm, res = 0.2, algorithm =  mba())
dtm_gbm <- grid_terrain(las.gbm, res = 0.2, algorithm =  mba())
dtm_knn <- grid_terrain(las.knn, res = 0.2, algorithm =  mba())
dtm_Ensamble <- grid_terrain(las.Ensamble.ground, res = 0.2, algorithm =  mba())
dtmActual <- grid_terrain(las, res = 0.2, algorithm =  mba())

plot_dtm3d(dtm_Ensamble, bg="white")
library(raster)
writeRaster(dtm_c50, "raster_output/")
writeRaster(dtm_svm, "raster_output/")
writeRaster(dtm_knn, "raster_output/")
writeRaster(dtm_gbm, "raster_output/")
writeRaster(dtm_Ensamble, "raster_output/")
writeRaster(dtmActual, "raster_output/")

### writelas
writeLAS(las.c50, "model_output/c50_classified.las" )
writeLAS(las.knn, "model_output/knn_classified.las" )
writeLAS(las.svm, "model_output/svm_classified.las" )
writeLAS(las.gbm, "model_output/gbm_classified.las" )
writeLAS(las.Ensamble, "model_output/Ensamble_classified.las" )
writeLAS(las, "model_output/actual_classified.las" )


### Cross Section

plot_crossection <- function(las,
                             p1 = c(min(las@data$X), mean(las@data$Y)),
                             p2 = c(max(las@data$X), mean(las@data$Y)),
                             width = 4, colour_by = NULL)
{
  colour_by <- enquo(colour_by)
  data_clip <- clip_transect(las, p1, p2, width)
  p <- ggplot(data_clip@data, aes(X,Z)) + geom_point(size = 0.5) + coord_equal() + theme_minimal()
  
  if (!is.null(colour_by))
    p <- p + aes(color = !!colour_by) + labs(color = "")
  
  return(p)
}

p1 = c(199392.48420057134,450849.2487765532)
p2 = c(199516.64906432584,450335.0519114015)
plot_crossection(las.Ensamble, colour_by = factor(Classification))

las.Ensamble.ground <- las.Ensamble[las.Ensamble$Classification == 2,]
las.svm.ground <- las.svm[las.svm$Classification == 2,]

plot(las.svm.ground)
