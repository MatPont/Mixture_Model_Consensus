setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(R.matlab)
library(mclust)
library(Rmixmod)
library(aricode)
library(movMF)
library(skmeans)



read_lil_matrix <- function(file_name){
  temp <- read.csv(file_name, header = FALSE)
  first_split_row <- unlist(strsplit(as.character(temp[1,]), " +", fixed = FALSE))
  nrow = as.integer(first_split_row[1])
  ncol = as.integer(first_split_row[2])
  mat <- matrix(0, nrow = nrow, ncol = ncol)
  for(i in 2:dim(temp)[1]){
    r_i <- i - 1
    split_row <- unlist(strsplit(as.character(temp[i,]), " +", fixed = FALSE))
    for(j in seq(2, length(split_row), 2)){
      mat[r_i, as.integer(split_row[j])] <- as.integer(split_row[j+1])
    }
  }
  return(mat)
}


#################################################
# Import data
#################################################
fbis <- read_lil_matrix("Dataset_CLUTO/fbis.mat")
fbis_y <- as.factor(as.integer(unlist(read.csv("Dataset_CLUTO/fbis.mat.rclass", header = FALSE))))

re0 <- read_lil_matrix("Dataset_CLUTO/re0.mat")
re0_y <- as.factor(as.integer(unlist(read.csv("Dataset_CLUTO/re0.mat.rclass", header = FALSE))))

tr23 <- read_lil_matrix("Dataset_CLUTO/tr23.mat")
tr23_y <- as.factor(as.integer(unlist(read.csv("Dataset_CLUTO/tr23.mat.rclass", header = FALSE))))

wap <- read_lil_matrix("Dataset_CLUTO/wap.mat")
wap_y <- as.factor(as.integer(unlist(read.csv("Dataset_CLUTO/wap.mat.rclass", header = FALSE))))

#################################################
# II
#################################################

run <- function(data, label, verbose=FALSE){
  
  #data <- type.convert(data)
  no_cluster <- length(unique(label))
  
  ### 1 ###
  cat("=======\n# movMF\n=======\n")
  res_models <- c()
  methods <- c("Banerjee_et_al_2005","Tanabe_et_al_2007","Sra_2012","Song_et_al_2012","uniroot","Newton","Halley","hybrid","Newton_Fourier")
  best_loglike <- NULL
  best_partition <- NULL
  for(method in methods){
    print(method)
    temp <- movMF(data, no_cluster, verbose=verbose, nruns=10, control = list(kappa = method))   
    res <- apply(temp$P, MARGIN=1, FUN=which.max)
    res_models <- cbind(res_models, res)
    if(is.null(best_loglike) || best_loglike < temp$L){
      best_loglike <- temp$L
      best_partition <- res
    }
  }
  cat("=======\n# Spherical k-means\n=======\n")
  temp <- skmeans(data, k = no_cluster)
  res_models <- cbind(res_models, temp$cluster)
  
  print(dim(res_models))
  
  ### 2 ###
  cat("=======\n# Best movMF\n=======\n")
  #temp <- movMF(data, no_cluster, verbose=verbose, nruns=10)   
  #res_vMF <- apply(temp$P, MARGIN=1, FUN=which.max)
  res_vMF <- best_partition
  
  #cat("=======\n# Best Gaussian\n=======\n")
  #res_mclust <- Mclust(data, G=no_cluster)$classification
  
  ### 3 ###
  cat("=======\n# Consensus\n=======\n")
  res_mixmod <- mixmodCluster(as.data.frame(type.convert(res_models)), nbCluster = no_cluster, models=mixmodMultinomialModel(), dataType="qualitative")
  
  #cat("##############\n# Best Gaussian\n##############\n")
  #cat("NMI: ", NMI(as.factor(res_mclust), as.factor(label)), "\n")
  #cat("ARI: ", ARI(as.factor(res_mclust), as.factor(label)), "\n")
  
  cat("##############\n# Default movMF\n##############\n")
  cat("NMI: ", NMI(as.factor(res_vMF), as.factor(label)), "\n")
  cat("ARI: ", ARI(as.factor(res_vMF), as.factor(label)), "\n")
  
  cat("##############\n# Consensus\n##############\n")
  cat("NMI: ", NMI(as.factor(res_mixmod@bestResult@partition), as.factor(label)), "\n")
  cat("ARI: ", ARI(as.factor(res_mixmod@bestResult@partition), as.factor(label)), "\n")
  
}

run(tr23, tr23_y)

run(fbis, fbis_y)

run(re0, re0_y)

run(wap, wap_y)
    

