## Reference XGB Implementation R
#https://www.kaggle.com/tqchen/otto-group-product-classification-challenge/understanding-xgboost-model-on-otto-data/notebook
library(xgboost)
setwd("C:\\Users\\jswain1.HOMEOFFICE\\Documents\\kaggle\\Give me Some Credit")


train<-read.csv("cs-training.csv")
test<-read.csv("cs-test.csv")
test$SeriousDlqin2yrs<-NULL ## To this Problem  Only ##(Drop Depedenet variable , as it is there is teseset)
## Missing value treatment #
sum(is.na(train)) ## Missing values train 
sum(is.na(test))

train[is.na(train)]<-999 ## replace missing values with 999
test[is.na(test)]<-999 ## replace missing values with 999

train[,2:11]<-log(train[,2:11]+1)
#train<-cbind(train,log(train[,2:11]+1))
test_vars<-names(test)
train_names<-names(train)
common_vars<-intersect(test_vars,train_names)
#all_data<-do.call(rbind,list(train[common_vars],test[common_vars]))
#unq_values<-data.frame(unique_values=apply(train,2,function(x) length(unique(x))))

## Data Transformation converting categorical features to integer#
#min_unq_values<-2

for (f in common_vars) {
  if (class(train[[f]]) == "character"){
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

######  END #######################


########## Event Rate Features ####
all_train<-xgb.DMatrix(data=data.matrix(train[common_vars]),label=(train[,c("SeriousDlqin2yrs")]))

#watchlist<-list(val=dval,train=dtrain)


#######  Get the Best Model Parameters ####

param <- list(objective           = "binary:logistic", 
              booster = "gbtree",
              eta                 = 0.01, # 0.06, #0.01,
              max_depth           = 20, #changed from default of 4,6,8,10,15,20
              subsample           = 0.5, #(.5,0.7,1)
              colsample_bytree    = 0.5, #(.5,0.7,1)
              min_child_weight=44.8833  ## 3/ Event rate - Rule of Thumb 
              
)

clf <- xgb.cv(params              = param, 
              data                = all_train, 
              nrounds             = 2000, #300, #280, #125, #250, # changed from 300
              verbose             = 1,
              #early.stop.round    = 40,
              #watchlist           = watchlist,
              maximize            = FALSE,
              eval_metric="auc",
              nfold=3
)

####################

###### Use the entire training set using best parameters 
clf_best <- xgboost(params        = param, 
                    data                = all_train, 
                    nrounds             = 674, #300, #280, #125, #250, # changed from 300
                    verbose             = 1,
                    #early.stop.round    = 200,
                    #watchlist           = watchlist,
                    maximize            = FALSE,
                    eval_metric="auc"
                    #nfold=3
                    
)

names(test)
test[,2:11]=log(test[,2:11]+1)

testDataMatrix<-xgb.DMatrix(data=data.matrix(test[,common_vars]))
pred1 <- predict(clf_best,testDataMatrix)
pred = matrix(pred1, nrow=1)
pred = data.frame(t(pred))
names(pred)<-"Probability"
pred$Id=test$id
write.csv(pred,file="XGB_Basic_log_transformation.csv",row.names=F)

###### Look at The Importance of Each variable ###
importance_matrix <- xgb.importance(common_vars, model = clf_best)