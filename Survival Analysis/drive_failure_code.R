### Drive Failure Analysis ###
# Data was compiled with python and stored in SQL db

## Read in libraries
library(dplyr)
library(ggplot2)
library(scales)
library(survival)


## Import/Clean Data
df <- read.csv("C:/Users/abtin/Documents/SQL/hd_failure2.csv")
df <- df |> filter(model != " 00MD00")


## Check for missing capacity
df |> filter(capacity == -1)


## Filling in missing capacity values
df <- df |>
  mutate(capacity = case_when(
    model == "WDC WUH722222ALE6L4" ~ 2.200097e+13,
    model == "WDC WUH721414ALE6L4" ~ 1.400052e+13,
    model == "ST12000NM0007" ~ 1.200014e+13,
    model == "ST16000NM001G" ~ 1.60009e+13,
    .default = capacity)
  )

df |> select(capacity) |> unique()
df |> filter(capacity %in% c(6.000000e+05,1.070000e+05)) #Checking outliers 
df <- df |> 
  mutate(capacity = capacity/(1e12)) |> #Convert bytes -> TB
  mutate(capacity = ifelse(
    capacity %in% c(6.000000e+05,1.070000e+05), 
    4.000000e+00,capacity)) #Fix incorrect capacity values


## Creating brand column
df |> select(model) |> unique() #All unique models


# Assigning brand names to each model
df <- df |>
  mutate(brand = case_when(
    apply(sapply(c("HGST","Hitachi"),grepl,model),1,any) ~ "HGST",
    apply(sapply(c("ST","Seagate"),grepl,model),1,any) ~ "Seagate",
    grepl("TOSHIBA",model,fixed=TRUE) ~ "Toshiba",
    apply(sapply(c("WD","WDC","WUH721816ALE6L4"),grepl,model),1,any) ~ "Western Digital",
    model == "CT250MX500SSD1" ~ "Crucial",
    grepl("DELL",model,fixed=TRUE) ~ "Dell",
    model == "HP SSD S700 250GB" ~ "HP",
    grepl("MTFD",model,fixed=TRUE) ~ "Micron",
    grepl("SAMSUNG",toupper(model),fixed=TRUE) ~ "Samsung",
    model %in% c("SSDSCKKB480G8R","SSDSCKKB240GZR") ~ "Intel",
    .default = NA))


## Visualize proportion of failures by top 4 brands
df |>
  filter(brand %in% c("Seagate","Toshiba","Western Digital","HGST")) |>
  ggplot(aes(y=brand,fill=factor(failure,labels=c("No","Yes")))) +
  geom_bar(aes(x=after_stat(count/sum(count)))) +
  scale_x_continuous(labels=percent) +
  labs(title="Drive Proportion by Brand and their Proportion of Failures",x="Proportion",y="Brand",fill = "Failure") +
  scale_fill_grey()


## Stratify capacity
df2 <- df |>
  filter(brand %in% c("Seagate","Toshiba","Western Digital","HGST")) |>
  mutate(capacity = factor(case_when(
    capacity < 8 ~ "[0,8)",
    capacity >= 8 & capacity < 12 ~ "[8-12)",
    capacity >= 12 & capacity < 16 ~ "[12-16)",
    capacity >= 16 ~ "[16,22]"),
    levels = c("[0,8)","[8-12)","[12-16)","[16,22]")))


## Fit KM estimates for capacity
fit.km <- survfit(Surv(observed_time,failure) ~ capacity, data=df2)

res <- summary(fit.km)
cols <- lapply(c(2:6, 8:11) , function(x) res[x])
dat <- do.call(data.frame,cols)

dat |>
  ggplot(aes(x=time,y=surv,color=strata)) +
  geom_step() +
  xlab("Time") +
  ylab("Survival Probability")

log.test <- survdiff(Surv(observed_time,failure) ~ capacity, data=df2)
log.test


## Fit KM estimates for brand
fit.km <- survfit(Surv(observed_time,failure) ~ brand, data=df2)

res <- summary(fit.km)
cols <- lapply(c(2:6, 8:11) , function(x) res[x])
dat <- do.call(data.frame,cols)

dat |>
  ggplot(aes(x=time,y=surv,color=strata)) +
  geom_step() +
  xlab("Time") +
  ylab("Survival Probability")

log.test <- survdiff(Surv(observed_time,failure) ~ brand, data=df2)
log.test