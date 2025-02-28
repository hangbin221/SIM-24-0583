library(lme4)
library(dhglm)

cwd = getwd()
dir_name = paste(cwd, '/', sep='')
model_names = c('GLMw', 'GLMM', 'HGLM')
data_names = c('epilepsy', 'cd4', 'bolus', 'owls', 'fruits')
epsilon=1e-8

# ========================= #
# ======= GLM ======= #
# ========================= #
model_name = 'PLM-fair'
msp   = matrix(NA, nrow=10, ncol=5)
msd   = matrix(NA, nrow=10, ncol=5)
r2p   = matrix(NA, nrow=10, ncol=5)
r2d   = matrix(NA, nrow=10, ncol=5)
times = matrix(NA, nrow=10, ncol=5)

for(data_number in 1:5){
  data_name = data_names[data_number]
  data = read.csv(paste(dir_name, data_name, '-prep.csv', sep=''))
  if("offset" %in% names(data)){
    data$y = data$y_raw
    data$offset = log(data$offset)
  }
  data$id = as.factor(data$id)
  x_cols <- names(data)[grepl("^x", names(data))]
  formula <- as.formula(paste("y ~", paste(x_cols, collapse = " + ")))
  for(fold_number in 0:9){
    data_trvl = data[data$fold!=fold_number,]
    data_trvl = data_trvl[data_trvl$fold!=(fold_number+1)%%10,]
    data_test = data[data$fold==fold_number,]
    start_time = Sys.time()
    if("offset" %in% names(data)){
      GLMw_res  = glm(formula, family = poisson, data = data_trvl, offset=offset)
    }else{
      GLMw_res  = glm(formula, family = poisson, data = data_trvl)
    }
    y_true = data_test$y
    y_pred = predict(GLMw_res, data_test, type='response')
    msp[  (fold_number+1),data_number] = mean((y_true-y_pred)^2/(y_pred+epsilon))
    msd[  (fold_number+1),data_number] = mean(y_true*log((y_true+epsilon)/(y_pred+epsilon))-(y_true-y_pred))
    numerator = mean((y_true-y_pred)**2/(y_pred+epsilon))
    denominator = mean((y_true-mean(y_true))**2/(mean(y_true)+epsilon))
    r2p[  (fold_number+1),data_number] = 1 - numerator / denominator
    numerator = mean(y_true*log((y_true+epsilon)/(y_pred+epsilon))-(y_true-y_pred))
    denominator = mean(y_true*log((y_true+epsilon)/(mean(y_true)+epsilon)))
    r2d[  (fold_number+1),data_number] = 1 - numerator / denominator
    times[(fold_number+1),data_number] = Sys.time() - start_time
  }
}
colnames(msp  ) = data_names
colnames(msd  ) = data_names
colnames(r2p  ) = data_names
colnames(r2d  ) = data_names
colnames(times) = data_names
write.csv(msp  , file=paste(dir_name, model_name, '-MSP-chosen.csv', sep=''), row.names=FALSE)
write.csv(msd  , file=paste(dir_name, model_name, '-MSD-chosen.csv', sep=''), row.names=FALSE)
write.csv(r2p  , file=paste(dir_name, model_name, '-R2P-chosen.csv', sep=''), row.names=FALSE)
write.csv(r2d  , file=paste(dir_name, model_name, '-R2D-chosen.csv', sep=''), row.names=FALSE)
write.csv(times, file=paste(dir_name, model_name, '-time-chosen.csv', sep=''), row.names=FALSE)


# ========================= #
# ======= GLMM ======= #
# ========================= #
model_name = 'PNLM-fair'
msp   = matrix(NA, nrow=10, ncol=5)
msd   = matrix(NA, nrow=10, ncol=5)
r2p   = matrix(NA, nrow=10, ncol=5)
r2d   = matrix(NA, nrow=10, ncol=5)
times = matrix(NA, nrow=10, ncol=5)

for(data_number in 1:5){
  data_name = data_names[data_number]
  data = read.csv(paste(dir_name, data_name, '-prep.csv', sep=''))
  if("offset" %in% names(data)){
    data$y = data$y_raw
    data$offset = log(data$offset)
  }
  data$id = as.factor(data$id)
  x_cols <- names(data)[grepl("^x", names(data))]
  formula <- as.formula(paste("y ~", paste(x_cols, collapse = " + "), " + (1|id)"))
  for(fold_number in 0:9){
    data_trvl = data[data$fold!=fold_number,]
    data_trvl = data_trvl[data_trvl$fold!=(fold_number+1)%%10,]
    data_test = data[data$fold==fold_number,]
    start_time = Sys.time()
    if("offset" %in% names(data)){
      GLMM_res  = glmer(formula, family = poisson, data = data_trvl, offset=offset)
      y_pred = predict(GLMM_res, data_test, type='response')*exp(data_test$offset)
    }else{
      GLMM_res  = glmer(formula, family = poisson, data = data_trvl)
      y_pred = predict(GLMM_res, data_test, type='response')
    }
    y_true = data_test$y
    msp[  (fold_number+1),data_number] = mean((y_true-y_pred)^2/(y_pred+epsilon))
    msd[  (fold_number+1),data_number] = mean(y_true*log((y_true+epsilon)/(y_pred+epsilon))-(y_true-y_pred))
    numerator = mean((y_true-y_pred)**2/(y_pred+epsilon))
    denominator = mean((y_true-mean(y_true))**2/(mean(y_true)+epsilon))
    r2p[  (fold_number+1),data_number] = 1 - numerator / denominator
    numerator = mean(y_true*log((y_true+epsilon)/(y_pred+epsilon))-(y_true-y_pred))
    denominator = mean(y_true*log((y_true+epsilon)/(mean(y_true)+epsilon)))
    r2d[  (fold_number+1),data_number] = 1 - numerator / denominator
    times[(fold_number+1),data_number] = Sys.time() - start_time
  }
}

colnames(msp  ) = data_names
colnames(msd  ) = data_names
colnames(r2p  ) = data_names
colnames(r2d  ) = data_names
colnames(times) = data_names
write.csv(msp  , file=paste(dir_name, model_name, '-MSP-chosen.csv', sep=''), row.names=FALSE)
write.csv(msd  , file=paste(dir_name, model_name, '-MSD-chosen.csv', sep=''), row.names=FALSE)
write.csv(r2p  , file=paste(dir_name, model_name, '-R2P-chosen.csv', sep=''), row.names=FALSE)
write.csv(r2d  , file=paste(dir_name, model_name, '-R2D-chosen.csv', sep=''), row.names=FALSE)
write.csv(times, file=paste(dir_name, model_name, '-time-chosen.csv', sep=''), row.names=FALSE)