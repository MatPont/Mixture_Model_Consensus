  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
  
  library(R.matlab)
  library(mclust)
  library(Rmixmod)
  library(aricode)
  
  
  #################################################
  # Import data
  #################################################
  jaffe <- readMat("Dataset/jaffe.mat")
  
  mnist <- readMat("Dataset/MNIST5.mat")
  
  mfea <- readMat("Dataset/MFEAT1.mat")
  
  usps <- readMat("Dataset/USPS.mat")
  
  optdigits <- readMat("Dataset/Optdigits.mat")
  
  
  #################################################
  # II
  #################################################
  
  run <- function(data, label){
    
    run_mclust <- function(data, modelNames = NULL, max_i = 10){
      best_loglik <- NULL
      best_res <- NULL    
      for(i in 1:max_i){
        res_temp <- try(Mclust(data, G = 10, modelNames = modelNames))
        if(typeof(res_temp) == "list"){
          if(is.null(best_loglik) || best_loglik < res_temp$loglik){
            best_loglik <- res_temp$loglik
            best_res <- res_temp
            print(best_loglik)
          }
        }
      }
      return(best_res)
    }
    
    #data <- type.convert(data)
    
    ### 1 ###
    res_models <- c()
    loglike <- c()
    bic <- c()
    for(model in mclust.options("emModelNames")){
      cat("=======\n# ", model, "\n=======\n")
      res_temp <- run_mclust(data, modelNames = model)
      if(typeof(res_temp) == "list"){
        to_add <- res_temp$classification
        res_models <- cbind(res_models, to_add)
        loglike <- c(loglike, res_temp$loglik)
        bic <- c(bic, res_temp$bic)
        print("OK!")
      }else{
        print("___KO!___")
      }
    }
    
    # Take only the best partitions according a criterion
    #order <- sort(loglike, decreasing = TRUE, index.return = TRUE)$ix[1:as.integer(length(loglike)/2)]
    order <- sort(bic, decreasing = TRUE, index.return = TRUE)$ix[1:as.integer(length(bic)/2)]
    #   res_models <- res_models[,order]
    print(dim(res_models))
    
    ### 2 ###
    cat("=======\n# Best Mclust\n=======\n")
    res_mclust <- run_mclust(data)
    
    ### 3 ###
    res_mixmod <- mixmodCluster(as.data.frame(type.convert(res_models)), nbCluster = 10, models=mixmodMultinomialModel(), dataType="qualitative")
    
    cat("##############\n# Best Mclust\n##############\n")
    cat("NMI: ", NMI(as.factor(res_mclust$classification), as.factor(label)), "\n")
    cat("ARI: ", ARI(as.factor(res_mclust$classification), as.factor(label)), "\n")
    
    cat("##############\n# Consensus\n##############\n")
    cat("NMI: ", NMI(as.factor(res_mixmod@bestResult@partition), as.factor(label)), "\n")
    cat("ARI: ", ARI(as.factor(res_mixmod@bestResult@partition), as.factor(label)), "\n")
    
    # for(i in 1:dim(res_models)[2]){
    #   print("=======")
    #   print(NMI(as.factor(res_models[,i]), as.factor(label)))
    #   print(ARI(as.factor(res_models[,i]), as.factor(label)))
    # }
  }
  
  dataset <- jaffe
  run(dataset$X, dataset$y)
    # ##############
    # # Best Mclust
    # ##############
    # NMI:  0.9540511 
    # ARI:  0.922681 
    # ##############  
    # # Consensus
    # ##############
    # NMI:  0.9623486 
    # ARI:  0.933061 
  
  dataset <- optdigits
  run(dataset$X, dataset$y)
  # ##############
  # # Best Mclust
  # ##############
  # NMI:  0.7430095 
  # ARI:  0.6602882     
  # ##############
  # # Consensus
  # ##############
  # NMI:  0.743212 
  # ARI:  0.6614359 
  
  dataset <- mfea
  run(dataset$X, dataset$y)
  
