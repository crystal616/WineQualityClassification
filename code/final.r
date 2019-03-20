rm(list=ls())

wine=read.csv('winequality-red.csv',sep=';')

names(wine)
table(wine$quality)
##   3   4   5   6   7   8 
##  10  53 681 638 199  18 

wine$levels=ifelse(wine$quality<=5,'Low','High')

varname=names(wine)[1:11]
pdf(file='C:\\Users\\YingCai\\Desktop\\st590\\wine1.pdf',paper='letter',height=10,width=8,title='Histograms of Train Data')
par(mfrow=c(3,2))
for(i in 1:11){
   hist(wine.train[,varname[i]],main=varname[i],xlab=varname[i])
}
dev.off()

pdf(file='C:\\Users\\YingCai\\Desktop\\st590\\wine2.pdf',paper='letter',height=10,width=8,title='Histograms of Test Data')
par(mfrow=c(3,2))
for(i in 1:11){
   hist(wine.test[,varname[i]],main=varname[i],xlab=varname[i])
}
dev.off()

pdf(file='C:\\Users\\YingCai\\Desktop\\st590\\wine2.pdf',paper='letter',height=10,width=8,title='Plots of predictors vs. quality')
par(mfrow=c(3,2))
for(i in c(1,3,5,7,10)){
   plot(x=subset(wine,levels=='Low')[,i],y=subset(wine,levels=='Low')[,i+1],col='green',xlab=varname[i],ylab=varname[i+1])
   points(x=subset(wine,levels=='High')[,i],y=subset(wine,levels=='High')[,i+1],col='red')
}
dev.off()

wine=wine[,-12]

set.seed(1)
train <- sample(dim(wine)[1], 1000)
wine.train <- wine[train, ]
wine.test <- wine[-train, ]
attach(wine)

table(wine.train$levels)
## High  Low 
##  516  484 
table(wine.test$levels)
## High  Low 
##  339  260 


#########################
##  Subset Selection   ##
#########################

###### Best Subset ######
library(ISLR)
library(leaps)
set.seed(2)
regfit.train=regsubsets(as.factor(levels)~.,wine.train,nvmax=11)
which.min(summary(regfit.train)$bic)
## [1] 5
summary(regfit.train)$which[5,]
## volatile.acidity, chlorides, total.sulfur.dioxide, sulphates, alcohol are TRUE.

######### Lasso #########
library(glmnet)
set.seed(2)
x=model.matrix(levels~.,wine)[,-1]
y0=wine$levels
y=ifelse(y0=='Low',0,1)
grid=10^seq(10,-2,length=100)
cv.out=cv.glmnet(x[train,],y[train],alpha=1)
bestlam=cv.out$lambda.min
lasso.out=glmnet(x[train,],y[train],alpha=1,lambda=grid)
lasso.coef=predict(lasso.out,type='coefficients',s=bestlam)[1:12,]
lasso.coef[lasso.coef!=0]
## fixed.acidity, volatile.acidity, chlorides, free.sulfur.dioxide, total.sulfur.dioxide, pH, sulphates, alcohol have non-zero estimated parameters.
## Total 8 predictors are chosen by Lasso.


#########################
## Logistic Regression ##
#########################

## Here I choose subset size=5 (volatile.acidity, chlorides, total.sulfur.dioxide, sulphates, alcohol)

set.seed(3)
glm.fit=glm(as.factor(levels)~volatile.acidity+chlorides+total.sulfur.dioxide+sulphates+alcohol,wine.train,family=binomial)
summary(glm.fit)
## The p-values of intercept, volatile.acidity, chlorides, total.sulfur.dioxide, sulphates and alcohol are very small (<0.001).
glm.probs=predict(glm.fit,wine.test,type='response')
glm.pred=rep('Low',dim(wine.test)[1])
glm.pred[glm.probs>0.5]='High'
table(prediction=glm.pred,true=wine.test$levels)
##           true
## prediction High Low
##       High   84 206
##      Low   255  54
## The error rate is 77.0%
## Logistic regression is not suitable for this case.


#########################
##         KNN         ##
#########################

library(class)
library(caret)
library(pROC)
set.seed(3)

best.k=train(levels~.,data=wine.train,method='knn',tuneGrid=expand.grid(.k=1:20),trControl=trainControl(method='repeatedcv',number=10,repeats=15,classProbs=T,summaryFunction=multiClassSummary))
## Choose K=13 based on the ROC, Accuracy, Sensitivity and Specificity. 
pdf(file='C:\\Users\\YingCai\\Desktop\\st590\\wine4.pdf',paper='letter',height=10,width=8,title='Choose K for KNN Classification')
plot(best.k,metric='ROC')
plot(best.k,metric='Accuracy')
plot(best.k,metric='Sensitivity')
plot(best.k,metric='Specificity')
dev.off()
set.seed(3)
knn.pred=knn(x[train,],x[-train,],y0[train],k=13,prob=T)
table(prediction=knn.pred,true=wine.test$levels)
##           true
## prediction High Low
##       High  236  87
##       Low   103 173
## The misclassification error is 31.7%, much better than logistic regression.


#########################
##         QDA         ##
#########################

library(MASS)
set.seed(3)
qda.fit=qda(levels~.,wine.train)
qda.class=predict(qda.fit,wine.test)$class
table(prediction=qda.class,true=wine.test$levels)
##           true
## prediction High Low
##       High  275  87
##       Low    64 173
## The misclassification error is 25.2%, better than KNN (K=13).


#########################
##    Pruned Tree      ##
#########################

library(tree)
tree.wine=tree(as.factor(levels)~.,wine.train)
set.seed(4)
cv.wine=cv.tree(tree.wine,FUN=prune.misclass)
cv.wine
## Best nodes = 8
prune.wine=prune.misclass(tree.wine,best=8)
tree.pred=predict(prune.wine,wine.test,type='class')
table(prediction=tree.pred,true=wine.test$levels)
##           true
## prediction High Low
##       High  225  39
##       Low   114 221
## The misclassification error is 25.5%
pdf(file='C:\\Users\\YingCai\\Desktop\\st590\\wine5.pdf',paper='a4r',height=10,width=8,title='Pruned Tree Classification')
plot(prune.wine)
text(prune.wine,pretty=0)
dev.off()


#########################
##    Random Forest    ##
#########################

library(randomForest)
set.seed(4)
rf.wine=randomForest(as.factor(levels)~.,data=wine.train,mtry=5,ntree=1423,importance=T)
rf.pred=predict(rf.wine,newdata=wine.test,type='response')
table(prediction=rf.pred,true=wine.test$levels)
##           true
## prediction High Low
##       High  276  36
##       Low    63 224
## The misclassification error is 16.5%, the smallest until now.


#########################
##       Boosting      ##
#########################

library(gbm)
set.seed(4)
wine10=data.frame(x,y)
boost.wine=gbm(y~.,data=wine10[train,],distribution='bernoulli',n.trees=10000,interaction.depth=3,shrinkage=0.001)
boost.prob=predict(boost.wine,newdata=wine10[-train,],n.trees=10000,type='response')
boost.pred=rep('Low',dim(wine.test)[1])
boost.pred[boost.prob>0.5]='High'
table(prediction=boost.pred,true=wine.test$levels)
##           true
## prediction High Low
##       High  265  49
##       Low    74 211
## The misclassification error is 20.5%, not too bad, but still higher than Random Forest.


################################
## SVM (radial an polynomial) ##
################################

library(e1071)
set.seed(4)
levels.factor=as.factor(wine$levels)
fac.wine=data.frame(wine[,-12],levels.factor)
tune.out=tune(svm,levels.factor~.,data=fac.wine[train,],kernel='radial',ranges=list(cost=c(0.1,1,10,100),gamma=c(0.2,0.5,1,2)))
summary(tune.out)
## The best choice is cost=1,gamma=0.5
table(prediction=predict(tune.out$best.model,newdata=wine.test),truth=wine.test$levels)
##           truth
## prediction High Low
##       High  262  42
##       Low    77 218
## The misclassification error is 19.9%, just a little higher than Random Forest.

set.seed(4)
tune.poly=tune(svm,levels.factor~.,data=fac.wine[train,],kernel='polynomial',ranges=list(cost=c(0.1,0.5,1,10),degree=c(2,3,4)))
summary(tune.poly)
## The best choice is cost=1,degree=3
table(prediction=predict(tune.poly$best.model,newdata=wine.test),truth=wine.test$levels)
##           truth
## prediction High Low
##       High  241  42
##       Low    98 218
## The misclassification error is 23.4%, not as good as Radial kernel SVM, and worse than Random Forest.





