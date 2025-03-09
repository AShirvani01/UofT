### AD Diagnosis Code ###
# Output was written in a csv for kaggle submission

## Read in libraries
library(dplyr)
library(caret)
library(ISLR)
library(glmnet)
library(class)
library(e1071)
library(xgboost)
library(catboost)
library(gtsummary)
set.seed(1)


## Import/Clean Data

train <- read.csv("~/train.csv")
test <- read.csv("~/test.csv")


datafmt <- function(data) {data |>
    mutate(PatientID = as.factor(PatientID)) |>
    mutate(Gender = as.factor(Gender)) |>
    mutate(Ethnicity = as.factor(Ethnicity)) |>
    mutate(EducationLevel = factor(EducationLevel, ordered = T)) |>
    mutate(Smoking = as.factor(Smoking)) |>
    mutate(FamilyHistoryAlzheimers = as.factor(FamilyHistoryAlzheimers)) |>
    mutate(CardiovascularDisease = as.factor(CardiovascularDisease)) |>
    mutate(Diabetes = as.factor(Diabetes)) |>
    mutate(Depression = as.factor(Depression)) |>
    mutate(HeadInjury = as.factor(HeadInjury)) |>
    mutate(Hypertension = as.factor(Hypertension)) |>
    mutate(MemoryComplaints = as.factor(MemoryComplaints)) |>
    mutate(BehavioralProblems = as.factor(BehavioralProblems)) |>
    mutate(Confusion = as.factor(Confusion)) |>
    mutate(Disorientation = as.factor(Disorientation)) |>
    mutate(PersonalityChanges = as.factor(PersonalityChanges)) |>
    mutate(DifficultyCompletingTasks = as.factor(DifficultyCompletingTasks)) |>
    mutate(Forgetfulness = as.factor(Forgetfulness)) |>
    select(!DoctorInCharge)
}

train <- train |>
  datafmt() |>
  select(!PatientID) |>
  mutate(Diagnosis = as.factor(Diagnosis))

test <- test |>
  datafmt()

x.train <- train |>
  select(!Diagnosis)

y.train <- train$Diagnosis

summary(train)
str(train)
paste("# of NULLs:",sum(is.na(train)))


## Correlation

cm <- cor(x.train |> 
            select(where(is.numeric) & !EducationLevel) |> 
            data.matrix(), 
          method = "pearson")
i <- which(row(cm)<col(cm), arr.ind=TRUE)


high_cor <- matrix(c(colnames(cm)[i],round(cm[i],3)),
                   ncol=3, 
                   dimnames=list(NULL,c("Feature 1", "Feature 2", "Correlation"))) |>
  as_tibble() |>
  mutate(Correlation = as.numeric(Correlation)) |>
  arrange(desc(abs(Correlation)))

head(high_cor,10) #Low Pearson & Spearman Correlation among numerical features


## Logistic Regression w/ Ridge/LASSO/Elastic net using glmnet

# Cross Validation for optimal lambda
cv.fits <- list()

for (i in seq(0,1,0.25)) {
  fit.name <- paste0("alpha",i)
  cv.fits[[fit.name]] <- cv.glmnet(x.train, 
                                   y.train, 
                                   family = "binomial",
                                   type.measure = "class",
                                   alpha = i) 
}

# Computing Predicted Diagnosis using optimal lambda (for each alpha)
for (i in seq(0,1,0.25)) {
  fit.name <- paste0("alpha",i)
  predicted <- predict(cv.fits[[fit.name]], 
                       s = cv.fits[[fit.name]]$lambda.min, 
                       newx = test |> as.matrix(), 
                       type = "class")
  
  temp <- data.frame(PatientID = test$PatientID, Diagnosis = predicted)
  colnames(temp)[2] <- "Diagnosis"
  write.csv(temp,paste0("~/predicted_alpha_",i,".csv"), row.names = F)
}


## K-Nearest Neighbours

for (i in seq(1,4,1)) {
  fit.name <- paste0("k",i)
  predicted <- knn(x.train,test, cl = y.train, k = i)
  
  temp <- data.frame(PatientID = test$PatientID, Diagnosis = predicted)
  write.csv(temp,paste0("~/predicted_knn",i,".csv"), row.names = F)
}


## Support Vector Machines

# Checking whether data is linear separable
svmfit <- svm(Diagnosis ~ ., data = train, kernel = "linear", cost = 1e10, scale = F)
predicted <- predict(svmfit, train |> select(!Diagnosis), type = "class")
mean(predicted == y.train) 

# Tuning cost parameter for linear kernel 
cv.svmfit <- tune.svm(Diagnosis ~ ., 
                      data = train, 
                      kernel = "linear", 
                      cost = 10^seq(-3,3))


# Tuning cost and gamma parameters for RBF kernel
cv.svmfit.r <- tune.svm(Diagnosis ~ ., 
                        data = train, 
                        kernel = "radial", 
                        cost = 10^seq(-3,4), 
                        gamma = 10^seq(-4,0))

svmfit.r <- svm(Diagnosis ~ ., data = train, kernel = "radial",
                gamma = cv.svmfit.r$best.parameters$gamma, 
                cost = cv.svmfit.r$best.parameters$cost,
                scale = T)

predicted.r <- predict(svmfit.r, newdata = test, type = "class")

temp <- data.frame(PatientID = test$PatientID, Diagnosis = predicted.r)
write.csv(temp, "~/SVM.csv", row.names = F)


## XgBoost

train_control <- trainControl(method = "cv", number = 5, search = "grid")

gbmGrid <- expand.grid(max_depth = c(3,4,5), 
                       nrounds = c(50,100,200),
                       eta = c(0.01,0.05,0.1,0.2,0.3),
                       gamma = c(0,0.25,1),
                       subsample = 1,
                       min_child_weight = 1,
                       colsample_bytree = 0.6)

model <- train(Diagnosis~., 
               data = train, 
               method = "xgbTree", 
               trControl = train_control, 
               tuneGrid = gbmGrid, 
               verbosity = 0)

predicted <- predict(model,test)
temp <- data.frame(PatientID = test$PatientID, Diagnosis = predicted)
write.csv(temp, "~/XGBoost.csv", row.names = F)


## CatBoost
start.time <- Sys.time()
# Hypertuning parameters
w <- ifelse(y.train==0,1,2)
train_control <- trainControl(method = "cv", number = 5, search = "grid")
catGrid <- expand.grid(depth = c(4,6,8,10,12),
                       learning_rate = c(0.01,0.02,0.03),
                       iterations = c(1000,1500,2000,3000),
                       l2_leaf_reg = c(5,8,12,15),
                       rsm = c(0.3,0.5,0.7),
                       border_count = c(128,160,192))

model <- train(x = x.train,
               y = y.train,
               method = catboost.caret,
               trControl = train_control,
               tuneGrid = catGrid,
               verbose = 1000,
               metric = "Logloss",
               weights = w)


model_imp <- varImp(model,scale=FALSE)[["importance"]]

temp <- tibble(Feature = row.names(model_imp),Importance = model_imp$Overall) |>
  arrange(Importance)

temp |>
  ggplot(aes(x=Importance,y=factor(Feature,levels=Feature))) +
  geom_bar(stat="identity") +
  labs(title = "Feature Importance",y = "Feature")

end.time <- Sys.time()
time.taken <- round(end.time - start.time)
paste("Runtime:",time.taken)

predicted <- predict(model,test)
temp <- data.frame(PatientID = test$PatientID, Diagnosis = predicted)
write.csv(temp, "~/CatBoost.csv", row.names = F)



train_pool <- catboost.load_pool(data = x.train, 
                                 label = y.train |> 
                                   as.matrix() |> 
                                   as.numeric())

model2 <- catboost.train(learn_pool = train_pool,
                         test_pool = NULL,
                         params = list(loss_function = "Logloss",
                                       depth = 10, 
                                       learning_rate = 0.01,
                                       iterations = 2000,
                                       l2_leaf_reg = 12,
                                       rsm = 0.5,
                                       border_count = 128,
                                       verbose = 1000))

catboost.get_feature_importance(model2)


## Summary Table

t1 <- train |> 
  mutate(Diagnosis = as.factor(ifelse(Diagnosis=="1","Positive","Negative"))) |>
  tbl_summary(include = c(ADL,BehavioralProblems,Diabetes,FunctionalAssessment,MemoryComplaints,SleepQuality),
              by = Diagnosis,
              label = c(BehavioralProblems = "Behavioral Problems",
                        FunctionalAssessment = "Functional Assessment",
                        MemoryComplaints = "Memory Complaints",
                        SleepQuality = "Sleep Quality")) |>
  modify_header(label = "**Feature**") |>
  modify_spanning_header(c("stat_1","stat_2") ~ "**Diagnosis**") |>
  add_p() |>
  bold_labels() |>
  filter_p(q=FALSE,t=0.05)