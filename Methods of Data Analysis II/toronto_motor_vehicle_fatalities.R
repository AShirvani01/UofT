### Identifying Risk Factors of Fatal MVCs Using Logistic Regression ###

#Importing Libraries

library(dplyr)
library(tidyr)
library(glmnet)
library(caTools)
library(rms)
library(pROC)
library(car)
library(ggplot2)
library(olsrr)
library(knitr)
library(gtsummary)


#Predictors of Interest: Speeding, IMPACTTYPE, INVAGE, PEDESTRIAN, ALCOHOL

#Read in dataset
df <- read.csv("~/datasets/toronto_motor_vehicle_collisions.csv", row.names=1)
df <- subset(df, select=-c(ACCNUM,DATE,STREET1,STREET2,OFFSET,DISTRICT,WARDNUM,
                           ACCLOC,INJURY,FATAL_NO,VEHTYPE,CYCLISTYPE,CYCACT,
                           CYCCOND,PEDTYPE,PEDACT,PEDCOND,MANOEUVER,DRIVCOND,
                           HOOD_140,NEIGHBOURHOOD_140,HOOD_158,DIVISION,geometry))
#drop date, assume independent of date/most of the variation informed by date is 
#encapsulated by other features (ie. weather)
#exact location is too difficult to incorporate, general location is already 
#encapsulated in intersection and neighborhood
#WARDNUM redundant, too similar to neighborhoods, neighborhoods more interpretable
#FATAL_NO missing too much info
#PED & CYC actions missing too much info
#ACCLOC conveys same information as LOCCOORD
#INJURY too similar to ACCLASS
#VEHTYPE already addressed by other features
#DRIVCOND already addressed by other features
#MANOEUVER too similar to DRIVACT
#Police division perfectly correlated with neighborhood


### Data Cleaning ###

#To reduce dimensionality, small subgroupings were grouped into "Other" category

df <- df |> 
  mutate(TIME = as.factor(case_when(
    TIME >= 0 & TIME < 600 ~ "Night",
    TIME >= 600 & TIME < 1200 ~ "Morning",
    TIME >= 1200 & TIME < 1800 ~ "Afternoon",
    TIME >= 1800 & TIME < 2400 ~ "Evening"))) |>
  mutate(RAMP = as.factor((case_when(
    grepl("Ramp",ROAD_CLASS,fixed=T) ~ "Yes",
    .default = "None")))) |>
  mutate(ROAD_CLASS = as.factor(case_when(
    ROAD_CLASS %in% c("None","Pending") ~ "Other",
    ROAD_CLASS == "Expressway Ramp" ~ "Expressway",
    ROAD_CLASS == "Major Arterial Ramp" ~ "Major Arterial",
    .default = ROAD_CLASS))) |> 
  mutate(LOCCOORD = as.factor(case_when(
    LOCCOORD %in% c("Intersection","Mid-Block") ~ LOCCOORD,
    .default = "Other"))) |> #Other = Exit/entrance ramps, park, private property, public lane, None
  mutate(TRAFFCTL = as.factor(case_when(
    TRAFFCTL %in% c("None","No Control") ~ "No Control",
    TRAFFCTL %in% c("Traffic Signal", "Stop Sign") ~ TRAFFCTL,
    .default = "Other"))) |> #Other = Pedestrian Crossover, Traffic controller, yield sign, school guard, police control, traffic gate, streetcar
  mutate(VISIBILITY = as.factor(case_when(
    VISIBILITY == "Drifting Snow" ~ "Snow",
    VISIBILITY == "Freezing Rain" ~ "Rain",
    VISIBILITY %in% c("None","Strong wind","Fog, Mist, Smoke, Dust") ~ "Other",
    .default = VISIBILITY))) |>
  mutate(ATFLIGHT = as.factor(case_when(
    grepl("artificial",LIGHT,fixed=T) ~ "Yes",
    .default = "None"))) |>
  mutate(LIGHT = as.factor(case_when(
    LIGHT == "Dark, artificial" ~ "Dark",
    LIGHT == "Dusk, artificial" ~ "Dusk",
    LIGHT == "Dawn, artificial" ~ "Dawn",
    LIGHT == "Daylight, artificial" ~ "Daylight",
    .default = LIGHT))) |> #create new feature for artificial light
  mutate(RDSFCOND = as.factor(case_when(
    RDSFCOND %in% c("Dry","Wet") ~ RDSFCOND,
    .default = "Other"))) |> #Other = Slush, loose snow, ice, packed snow, sand/gravel, none, other
  mutate(ACCLASS = relevel(as.factor(case_when(
    ACCLASS == "Fatal" ~ "Fatal",
    TRUE ~ "Non-Fatal")),"Non-Fatal")) |>
  mutate(IMPACTYPE = as.factor(case_when(
    grepl("SMV",IMPACTYPE,fixed=T) ~ "SMV",
    IMPACTYPE == "None" ~ "Other",
    .default = IMPACTYPE))) |> #Angle collisions are the most common fatal type in canada
  mutate(INVTYPE = as.factor(case_when(
    grepl("Driver",INVTYPE,fixed=T) ~ "Driver",
    grepl("Passenger",INVTYPE,fixed=T) ~ "Passenger",
    grepl("Pedestrian",INVTYPE,fixed=T) ~ "Pedestrian",
    .default = "Other"))) |> #Other = cyclist, property owner, trailer owner, witness, vehicle owner, other
  mutate(INVAGE = factor(case_when(
    INVAGE %in% c("0 to 4","5 to 9","10 to 14","15 to 19") ~ "Adolescence",
    INVAGE %in% c("20 to 24","25 to 29","30 to 34","35 to 39") ~ "Early Adulthood",
    INVAGE %in% c("40 to 44","45 to 49","50 to 54","55 to 59") ~ "Mid Adulthood",
    INVAGE %in% c("60 to 64","65 to 69","70 to 74","75 to 79",
                  "80 to 84","85 to 89","90 to 94","Over 95") ~ "Late Adulthood",
    INVAGE == "unknown" ~ "Unknown"),
    levels = c("Adolescence","Early Adulthood","Mid Adulthood","Late Adulthood","Unknown"))) |>
  mutate(INITDIR = as.factor(case_when(
    INITDIR == "None" ~ "Unknown",
    .default = INITDIR))) |>
  mutate(DRIVACT = as.factor(DRIVACT)) |>
  mutate(PEDESTRIAN = as.factor(PEDESTRIAN)) |>
  mutate(CYCLIST = as.factor(CYCLIST)) |>
  mutate(AUTOMOBILE = as.factor(AUTOMOBILE)) |>
  mutate(MOTORCYCLE = as.factor(MOTORCYCLE)) |>
  mutate(TRUCK = as.factor(TRUCK)) |>
  mutate(TRSN_CITY_VEH = as.factor(TRSN_CITY_VEH)) |>
  mutate(EMERG_VEH = as.factor(EMERG_VEH)) |>
  mutate(PASSENGER = as.factor(PASSENGER)) |>
  mutate(SPEEDING = as.factor(SPEEDING)) |>
  mutate(AG_DRIV = as.factor(AG_DRIV)) |>
  mutate(REDLIGHT = as.factor(REDLIGHT)) |>
  mutate(ALCOHOL = as.factor(ALCOHOL)) |>
  mutate(DISABILITY = as.factor(DISABILITY)) |>
  mutate(NEIGHBOURHOOD_158 = as.factor(NEIGHBOURHOOD_158))

str(df)
sum(is.na(df))


### EDA ###

t1 <- df %>%
  tbl_summary(include = c(SPEEDING,IMPACTYPE,INVAGE,PEDESTRIAN,ALCOHOL),
              label = c(SPEEDING ~ "Speeding-related",
                        IMPACTYPE ~ "Type of Impact",
                        INVAGE ~ "Age Group of Those Involved",
                        PEDESTRIAN ~ "Pedestrian Involved",
                        ALCOHOL ~ "Alcohol-related"),
              sort = IMPACTYPE ~ "frequency",
              by = ACCLASS) %>%
  add_overall(last = TRUE, col_label = "**Total**", statistic = ~ "{n}") %>%
  add_p() %>%
  modify_header(label ~ "**Risk Factor**") %>%
  modify_spanning_header(c("stat_1","stat_2") ~ "**Accident Class**") %>%
  modify_caption("**Table 1. Distribution of fatal and non-fatal MVCs by risk factor**") %>%
  bold_labels() %>%
  modify_footnote(c(stat_1,stat_2) ~ "Count (%)",
                  stat_0 ~ "Count")

gt::gtsave(as_gt(t1), file = file.path(tempdir(), "t1.png"))


### Variable Selection ###

#Each accident is assumed to be independent of one another

ST <- Sys.time()
glm.mod1 <- glm(ACCLASS ~ ., family = binomial(link = logit), data = df)
Sys.time() - ST

ST <- Sys.time()
#AIC
sel.var.aic <- step(glm.mod1, trace = 0, k = 2, direction = "both") 
select_var_aic<-attr(terms(sel.var.aic), "term.labels") 
Sys.time() - ST


#BIC
ST <- Sys.time()
sel.var.bic <- step(glm.mod1, trace = 0, k = log(nrow(df)), direction = "both") 
select_var_bic<-attr(terms(sel.var.bic), "term.labels")
Sys.time() - ST


#LASSO/Elastic net
set.seed(1)

x <- model.matrix( ~ ., df[,which(colnames(df) != "ACCLASS")])
y <- df$ACCLASS

cvfit <- cv.glmnet(x, y, family = "binomial", type.measure = "class") #Minimize misclassification error
#plot(cvfit)
#y_pred <- predict(cvfit, newx = x, type = "class",s = cvfit$lambda.min)
#misclasserror <- mean(y_pred != y)

coef(cvfit, s = "lambda.min")

select_var_las <- colnames(df[, -which(colnames(df) %in% c("RDSFCOND","INVTYPE","PEDESTRIAN","AUTOMOBILE","EMERG_VEH","DISABILITY","ACCLASS"))])



### Model validation & Calibration ###

x11(width = 10, height = 10)
layout(mat = matrix(c(1,3,5,2,4,6),nrow = 3, ncol = 2))

#AIC
lrm.aic <- lrm(ACCLASS ~ ., data = df[,which(colnames(df) %in% c(select_var_aic, "ACCLASS"))], x =TRUE, y = TRUE, model= T)
cross.calib <- calibrate(lrm.aic, method="crossvalidation", B=10) # model calibration
plot(cross.calib, las=1, main = "AIC Calibration Plot", xlab = "Predicted Probability", cex.sub = 0.85) #Calibration plot

p.aic <- predict(lrm.aic, type = "fitted")

roc.aic <- roc(df$ACCLASS ~ p.aic)
TPR <- roc.aic$sensitivities #True Positive Rate
FPR <- 1 - roc.aic$specificities #False Positive Rate

plot(FPR, TPR, main = "AIC ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate", xlim = c(0,1), ylim = c(0,1), type = 'l', lty = 1, lwd = 2,col = 'red') #ROC curve
abline(a = 0, b = 1, lty = 2, col = 'blue')
text(0.7,0.4,label = paste("AUC = ", round(auc(roc.aic),2)))


#BIC
lrm.bic <- lrm(ACCLASS ~ ., data = df[,which(colnames(df) %in% c(select_var_bic, "ACCLASS"))], x =TRUE, y = TRUE, model= T)
cross.calib <- calibrate(lrm.bic, method="crossvalidation", B=10) # model calibration
plot(cross.calib, las=1, main = "BIC Calibration Plot", xlab = "Predicted Probability", cex.sub = 0.85) #Calibration plot

p.bic <- predict(lrm.bic, type = "fitted")

roc.bic <- roc(df$ACCLASS ~ p.bic)
TPR <- roc.bic$sensitivities #True Positive Rate
FPR <- 1 - roc.bic$specificities #False Positive Rate

plot(FPR, TPR, main = "BIC ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate", xlim = c(0,1), ylim = c(0,1), type = 'l', lty = 1, lwd = 2,col = 'red') #ROC curve
abline(a = 0, b = 1, lty = 2, col = 'blue')
text(0.7,0.4,label = paste("AUC = ", round(auc(roc.bic),2)))



#LASSO
lrm.las <- lrm(ACCLASS ~ ., data = df[,which(colnames(df) %in% c(select_var_las, "ACCLASS"))], x =TRUE, y = TRUE, model= T)
cross.calib <- calibrate(lrm.las, method="crossvalidation", B=10) # model calibration
plot(cross.calib, las=1, main = "LASSO Calibration Plot", xlab = "Predicted Probability", cex.sub = 0.85) #Calibration plot

p.las <- predict(lrm.las, type = "fitted")

roc.las <- roc(df$ACCLASS ~ p.las)
TPR <- roc.las$sensitivities #True Positive Rate
FPR <- 1 - roc.las$specificities #False Positive Rate

plot(FPR, TPR, main = "LASSO ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate", xlim = c(0,1), ylim = c(0,1), type = 'l', lty = 1, lwd = 2,col = 'red') #ROC curve
abline(a = 0, b = 1, lty = 2, col = 'blue')
text(0.7,0.4,label = paste("AUC = ", round(auc(roc.las),2)))

while(names(dev.cur()) !='null device') Sys.sleep(1)


### Diagnostics for BIC model ###

df.bic <- df[,which(colnames(df) %in% select_var_bic)]

glm.bic <- glm(ACCLASS ~ ., family = binomial, data = df[,which(colnames(df) %in% c(select_var_bic, "ACCLASS"))])
df.final <- dfbetas(glm.bic)

for (x in 1:ncol(df.bic)) {
  par(family = 'serif')
  plot(1:nrow(df.bic), df.final[,x], xlab="Observation Index", 
       ylab='dfbeta')
  title(colnames(df.bic)[x])
  lines(lowess(df[,x], df.final[,x]), lwd=2, col='blue')
  abline(h=0, lty='dotted')
  abline(h=-2/sqrt(nrow(df.final)), lty='dotted')
  abline(h=2/sqrt(nrow(df.final)), lty='dotted')
}

ols_plot_cooksd_chart(glm.bic, type = 2)
#Observations to check: 17686, 12410,12420,8154,8155

vif(glm.bic) #Pedestrian + Cyclist high adjusted GVIF -> multicollinearity


### Refit BIC model ###

x11(width = 10, height = 10)
layout(mat = matrix(c(1:2),nrow = 1, ncol = 2))

df.bic2 <- df[,which(colnames(df) %in% select_var_bic & colnames(df) != c("PEDESTRIAN","CYCLIST"))]
glm.bic2 <- glm(ACCLASS ~ ., family = binomial, data = df[,which(colnames(df) %in% c(select_var_bic, "ACCLASS") & colnames(df) != c("PEDESTRIAN","CYCLIST"))])

lrm.bic2 <- lrm(ACCLASS ~ ., data = df[,which(colnames(df) %in% c(select_var_bic, "ACCLASS") & colnames(df) != c("PEDESTRIAN","CYCLIST"))], x =TRUE, y = TRUE, model= T)
cross.calib <- calibrate(lrm.bic2, method="crossvalidation", B=10) # model calibration
plot(cross.calib, las=1, xlab = "Predicted Probability") #Calibration plot

p.bic2 <- predict(lrm.bic2, type = "fitted")

roc.bic2 <- roc(df$ACCLASS ~ p.bic2)
TPR <- roc.bic2$sensitivities #True Positive Rate
FPR <- 1 - roc.bic2$specificities #False Positive Rate

plot(FPR, TPR, xlim = c(0,1), ylim = c(0,1), type = 'l', lty = 1, lwd = 2,col = 'red') #ROC curve
abline(a = 0, b = 1, lty = 2, col = 'blue')
text(0.7,0.4,label = paste("AUC = ", round(auc(roc.bic2),2)))

while(names(dev.cur()) !='null device') Sys.sleep(1)
#AUC same, Calibration plot better fit (MAE/MSE much lower)

lrtest(glm.bic, glm.bic2)


### Final model ###

t3 <- glm.bic2 %>%
  tbl_regression(exponentiate = TRUE,
                 show_single_row = c("MOTORCYCLE","TRUCK","TRSN_CITY_VEH","PASSENGER","SPEEDING","AG_DRIV","REDLIGHT","ATFLIGHT"),
                 label = c(YEAR ~ "Year",
                           VISIBILITY ~ "Weather Condition",
                           LIGHT ~ "Level of Light",
                           IMPACTYPE ~ "Type of Impact",
                           INVAGE ~ "Age of Those Involved",
                           MOTORCYCLE ~ "Motorcycle Involved",
                           TRUCK ~ "Truck Involved",
                           TRSN_CITY_VEH ~ "City Transit Vehicle Involved",
                           PASSENGER ~ "Passenger involved",
                           SPEEDING ~ "Speeding-related",
                           AG_DRIV ~ "Aggressive Driving-related",
                           REDLIGHT ~ "Redlight-related",
                           ATFLIGHT ~ "Artificial light")) %>%
  modify_header(label ~ "**Risk Factor**", estimate ~ "**AOR**") %>%
  modify_caption("**Table 3. Final Logistic regression model results, representing the odds of fatality in a MVC in Toronto from 2006-2021.**") %>%
  bold_labels() %>%
  modify_footnote(c(everything() ~ NA, estimate ~ "AOR = Adjusted Odds Ratio", ci ~ "CI = Confidence Interval"), abbreviation = TRUE) %>%
  bold_p()

gt::gtsave(as_gt(t3), file = file.path(tempdir(), "t3.png"))
