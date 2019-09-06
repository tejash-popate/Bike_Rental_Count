rm(list = ls())
path="C:/Users/ELdrago/Desktop/Edwisor/Projects/Bike Rental Count/R_Code/Input_files"
#setwd(path)


#-------load required libraries
x=c("tidyverse", "corrgram", "DMwR", "caret", "randomForest", "unbalanced",
    "C50", "dummies", "e1071", "Information","MASS", "rpart", "gbm", "ROSE",'car','Hmisc','funModeling',
  'DataCombine', 'inTrees','missMDA','data.table','lsr',"usdm")
lapply(x,require,character.only= TRUE) 
rm(x)

#load the dataset
bike_data=read.csv("C:/Users/ELdrago/Desktop/Edwisor/Projects/Bike Rental Count/R_Code/Input_files/day.csv")

View(bike_data)
str(bike_data)
#-------------------------------------------
#-------------------1)Exploratory Data Analysis
cont_vars =c('temp','atemp','hum','windspeed','casual','registered','cnt')

cat_vars = c('season','yr','mnth','holiday','weekday','workingday','weathersit')
categorical_data=subset(bike_data,select=cat_vars)
continuous_data=subset(bike_data,select=cont_vars)
target_var='cnt'



# Univariate Analysis
num_eda <- function(data)
{
  glimpse(data)
  df_status(data)
  profiling_num(data)
  plot_num(data)
  describe(data)
}
cat_eda <- function(data)
{
  glimpse(data)
  df_status(data)
  freq(data) 
  plot_num(data)
  describe(data)
}

cat_eda(categorical_data)
num_eda(continuous_data)

#Check the distribution of numerical data using scatterplot
scat1 = ggplot(data = bike_data, aes(x =temp, y = cnt)) + ggtitle("Distribution of Temperature") + geom_point() + xlab("Temperature") + ylab("Bike COunt")
scat2 = ggplot(data =bike_data, aes(x =hum, y = cnt)) + ggtitle("Distribution of Humidity") + geom_point(color="red") + xlab("Humidity") + ylab("Bike COunt")
scat3 = ggplot(data = bike_data, aes(x =atemp, y = cnt)) + ggtitle("Distribution of Feel Temperature") + geom_point() + xlab("Feel Temperature") + ylab("Bike COunt")
scat4 = ggplot(data = bike_data, aes(x =windspeed, y = cnt)) + ggtitle("Distribution of Windspeed") + geom_point(color="red") + xlab("Windspeed") + ylab("Bike COunt")
gridExtra::grid.arrange(scat1,scat2,scat3,scat4,ncol=2)

# Give the chart file a name.
png(file = "scatterplot_matrices.png")
# Plot the matrices between 4 variables giving 12 plots.
# One variable with 3 others and total 4 variables.
pairs(~temp+hum+atemp+windspeed,data = bike_data,col="blue",
      main = "Scatterplot Matrix")
# Save the file.
dev.off()



#-----------------2) Missing Value Analysis
sum(is.na(bike_data))
#----no missing values

#---------------3) Outlier Analysis
for (i in 1:length(cont_vars))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cont_vars[i]), x = target_var), data = subset(bike_data))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=10,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cont_vars[i],x=target_var)+
           ggtitle(paste("Box plot of",cont_vars[i])))
}
# plotting plots together
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,ncol=2)
gridExtra::grid.arrange(gn7,ncol=1)
# removing outliers using boxplot method
df=copy(bike_data) # for backup
for(i in cont_vars){
  print(i)
  val=bike_data[,i][bike_data[,i] %in% boxplot.stats(bike_data[,i])$out ]
  bike_data=bike_data[which(!bike_data[,i] %in% val),]
}
str(bike_data)
# replacing outliers with NA and use KNN imputation
for(i in cont_vars){
  print(i)
  val=bike_data[,i][bike_data[,i] %in% boxplot.stats(bike_data[,i])$out ]
  bike_data[,i][(bike_data[,i] %in% val)]= NA
}
sum(is.na(bike_data))
#KNN imputation
bike_data=knnImputation(bike_data, k=5)

#----------------4) Feature Selection
#Check for multicollinearity using VIF
vifcor(bike_data[,c("temp","atemp","hum","windspeed")])

# Correlation plot
jpeg("correlation_plot.jpg", width = 350, height = 350)
corrgram(bike_data[,cont_vars],order=F,upper.panel = panel.pie,text.panel = panel.txt, main="Correlation Plot")
dev.off()

# ANOVA test for categorical variables
library("lsr")
anova_test = aov(cnt~season+yr+mnth+weekday+workingday+weathersit, data = bike_data)
summary(anova_test)

# dimension reduction
bike_data = subset(bike_data, select = -c(instant,casual,registered,temp,dteday))
cleaned_data=copy(bike_data)
write.csv(cleaned_data,"cleaned_data_by_tejashpopate.csv",row.names=F)
#----------------5) Feauture Scaling

# Creating dummy variables for categorical variables
bike_data = dummy.data.frame(bike_data, cat_vars)

#--------------------MODEL DEVELOPEMENT

rmExcept("bike_data")

# splitting the dataset
#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index=sample(1:nrow(bike_data),0.8*nrow(bike_data))
#train.index=createDataPartition(bike_data$Absenteeism_time_in_hours,p=.80,list=F)
train_data=bike_data[train.index,]
test_data=bike_data[-train.index,]
# r squared evaluation and function for MAPE

MAPE<-function(predicted,actual) {
  mape<-mean(abs((actual-predicted)/actual) * 100)
  return(mape)
  }
rsq <- function(x, y) summary(lm(y~x))$r.squared
target_var='cnt'

#----------------------- 1) Linear Regression

set.seed(1234)
#Develop Model on training data
fit_LR = lm(cnt ~ ., data = train_data)
#write summary into disk
write(capture.output(summary(fit_LR)), "LR_summary.txt")
#Lets predict for training data
pred_LR_train = predict(fit_LR, train_data[,names(test_data) != target_var])
#Lets predict for testing data
pred_LR_test = predict(fit_LR,test_data[,names(test_data) != target_var])
# For training data 
print(regr.eval(train_data[,target_var],pred_LR_train,stats=c("rmse","mse","mae")))
cat("rsq for train", rsq(train_data[,target_var],pred_LR_train),"\nMAPE for train",MAPE(pred_LR_train,train_data$cnt))
# For testing data 
print(regr.eval(test_data[,target_var],pred_LR_test,stats=c("rmse","mse","mae")))
cat("rsq for test", rsq(test_data[,target_var],pred_LR_test),"\nMAPE for test",MAPE(pred_LR_test,test_data$cnt))

#------------------------2) Decision Trees
set.seed(1234)
#Develop Model on training data
fit_DT = rpart(cnt ~., data = train_data, method = "anova")
#Summary of DT model
summary(fit_DT)
#write rules into disk
write(capture.output(summary(fit_DT)), "DT_Rules.txt")
#Lets predict for training data
pred_DT_train = predict(fit_DT, train_data[,names(test_data) != target_var])
#Lets predict for training data
pred_DT_test = predict(fit_DT,test_data[,names(test_data) != target_var])
# For training data 
print(regr.eval(train_data[,target_var],pred_DT_train,stats=c("rmse","mse","mae")))
cat("rsq for train", rsq(train_data[,target_var],pred_DT_train),"\nMAPE for train",MAPE(pred_DT_train,train_data$cnt))
# For testing data 
print(regr.eval(test_data[,target_var],pred_DT_test,stats=c("rmse","mse","mae")))
cat("rsq for test", rsq(test_data[,target_var],pred_DT_test),"\nMAPE for test",MAPE(pred_DT_test,test_data$cnt))

#-------------------------3) Random Forest
set.seed(1234)
#Develop Model on training data
fit_RF = randomForest(cnt~., data = train_data,)
#write summary into disk
write(capture.output(summary(fit_RF)), "RF_summary.txt")
#Lets predict for training data
pred_RF_train = predict(fit_RF, train_data[,names(test_data) != target_var])
#Lets predict for testing data
pred_RF_test = predict(fit_RF,test_data[,names(test_data) != target_var])
# For training data 
print(regr.eval(train_data[,target_var],pred_RF_train,stats=c("rmse","mse","mae")))
cat("rsq for train", rsq(train_data[,target_var],pred_RF_train),"\nMAPE for train",MAPE(pred_RF_train,train_data$cnt))
# For testing data 
print(regr.eval(test_data[target_var],pred_RF_test,stats=c("rmse","mse","mae")))
cat("rsq for test", rsq(test_data[,target_var],pred_RF_test),"\nMAPE for test",MAPE(pred_RF_test,test_data$cnt))

#-------------------------4) XGBoost
#Develop Model on training data
fit_XGB = gbm(cnt~., data = train_data, n.trees =500, interaction.depth = 2)
#Lets predict for training data
pred_XGB_train = predict(fit_XGB, train_data, n.trees = 500)
#write summary into disk
write(capture.output(summary(fit_LR)), "XGB_summary.txt")
#Lets predict for testing sdata
pred_XGB_test = predict(fit_XGB,test_data, n.trees = 500)
# For training data 
print(regr.eval(train_data[target_var],pred_XGB_train,stats=c("rmse","mse","mae")))
cat("rsq for test", rsq(train_data[,target_var],pred_XGB_train),"\nMAPE for test",MAPE(pred_XGB_train,train_data$cnt))
# For testing data 
print(regr.eval(test_data[target_var],pred_XGB_test,stats=c("rmse","mse","mae")))
cat("rsq for test", rsq(test_data[,target_var],pred_XGB_test),"\nMAPE for test",MAPE(pred_XGB_test,test_data$cnt))
#----------------------Visualize fit of models
scatterplot(train_data$cnt~pred_LR_train)
scatterplot(test_data$cnt~pred_LR_test)
scatterplot(train_data$cnt~pred_DT_train)
scatterplot(test_data$cnt~pred_DT_test)
scatterplot(train_data$cnt~pred_RF_train)
scatterplot(test_data$cnt~pred_RF_test)
scatterplot(train_data$cnt~pred_XGB_train)
scatterplot(test_data$cnt~pred_XGB_test)
#---------------------------End of the Project----------------------------#
