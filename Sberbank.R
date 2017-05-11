library(data.table)
library(xgboost)
library(Metrics)

setwd <- "C:/R-Studio/Sberbank"
train  <- fread( "./train.csv")
test  <- fread( "./test.csv")
macro <- fread("./macro.csv")

str(train)
ncol(train)

feature.names <- setdiff(colnames(train),c("price_doc","id","timestamp"))
y <- train$price_doc
characterVars <- names(which(sapply(train, class)=='character'))
noCharacter <- length(names(which(sapply(train, class)=='character')))
noInteger <- length(names(which(sapply(train, class)=='integer')))
noNumeric <- length(names(which(sapply(train, class)=='numeric')))
noAllCol <- noCharacter + noInteger + noNumeric

# convert all character cols to numeric (integer encoding)
cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in characterVars) {
  colValues <- c(train[[f]],test[[f]])
  uniquecolValues <- table(colValues)
  levels <- names(uniquecolValues[order(uniquecolValues)])
  train[[f]] <- as.integer(factor(train[[f]], levels=levels))
  test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
}

custom_rmsle <- function(preds, dtrain) {
  labels = dtrain.get_label()
  err <- rmsle(labels,preds)
  list(metric='rmsle',value=err)
}

xgb_params = list(eta= 0.05,max_depth= 5,subsample= 0.7,colsample_bytree=0.7,objective='reg:linear',
                  feval='custom_rmsle',min_child_weight = 1,maximize=FALSE)

xtrainAll <- xgb.DMatrix(data.matrix(train[,feature.names,with=FALSE]), label=y, missing=NA)
xgboost.cv.fit <- xgb.cv (data=xtrainAll,params=xgb_params,nround=500, metrics=list('rmse'),
                           early_stopping_rounds = 20, print_every_n =50,nfold=5)
# test-rmse 2623456

watchlist=list(xtrain=xtrainAll)
xgboost.fit <- xgb.train (data=xtrainAll,params=xgb_params,nround=xgboost.cv.fit$best_iteration, 
                             print_every_n =5,watchlist=watchlist)

importance_matrix <- xgb.importance(model = xgboost.fit, feature.names)
head(importance_matrix,30)
train$preds <- predict(xgboost.fit,xtrainAll)
train$preds <- ifelse(train$preds < 0 , 0,train$preds)
rmsle(train$price_doc,train$preds )

sample_submission <- fread("./sample_submission.csv")
xtest <- xgb.DMatrix(data.matrix(test[,feature.names,with=FALSE]), missing=NA)
preds_test <- predict(xgboost.fit,xtest)
sample_submission$price_doc <- preds_test
write.csv(sample_submission,file = "Sberbank submission.csv",row.names = F)

