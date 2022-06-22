---
  title: "Heart Failure"
author: "Jeremy Larcher"
date: '2022-06-22'
output: html_document
---
  
  ```{r Libraries}
library(tidyr)
library(ggplot2)
library(corrr)
library(rsample)
library(recipes)
library(parsnip)
library(yardstick)
library(skimr)
library(psych)
library(ranger)
library(tidyverse)
library(tidymodels)
library(GGally)
options(scipen=999)
library(doParallel)
library(themis)
```

```{r EDA & Cleaning}
skim(heart)

heart %>% 
  count(HeartDisease)

heart$HeartDisease <- as.numeric(heart$HeartDisease)

heart <- heart %>% 
  mutate(HeartDisease = case_when(HeartDisease > 0.5 ~ "Yes", TRUE ~ "No"))


heart$HeartDisease <- as.factor(heart$HeartDisease)
heart$FastingBS <- as.character(heart$FastingBS)


## Numerical Plots
heart %>% 
  select(HeartDisease, Age, RestingBP, Cholesterol, MaxHR, Oldpeak) %>% 
  ggpairs(columns = 2:6, aes(color = HeartDisease, alpha = 0.5))

## Categorical Plots

heart %>% 
  select(HeartDisease, Sex, ChestPainType,RestingECG,ExerciseAngina,FastingBS,ST_Slope) %>% 
  pivot_longer(Sex:ST_Slope) %>% 
  ggplot(aes(y = value, fill = HeartDisease))+
  geom_bar(position = "fill")+
  facet_wrap(vars(name), scale = "free")+
  labs(x = NULL, y= NULL, fil = NULL)


```

```{r Splitting Data}
set.seed(123)
data_split <- initial_split(heart)
data_train <- training(data_split)
data_test <- testing(data_split)
```

```{r Folds}
heart_folds <- vfold_cv(data_train, v = 5, strata = HeartDisease)
heart_folds

heart_metrics <- metric_set(accuracy, sensitivity, specificity, recall)
```

```{r Recipe}
heart_recipe <- recipe(HeartDisease ~ ., data = data_train) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(), - all_outcomes()) %>% 
  step_zv(all_predictors())

heart_recipe
```

```{r Bag Tree Model}

library(baguette)

bag_spec <-
  bag_tree(min_n = 10) %>% 
  set_engine("rpart", times = 25) %>% 
  set_mode("classification")

```

```{r Fitting Model to Data}
imb_wf <- workflow() %>% 
  add_recipe(heart_recipe) %>% 
  add_model(bag_spec)

fit(imb_wf, data = data_train)
```

```{r Class Imbalance}
doParallel::registerDoParallel()
set.seed(123)
imb_results <- fit_resamples(
  imb_wf,
  resamples = heart_folds,
  metrics = heart_metrics
)

collect_metrics(imb_results)
```

```{r Balancing}
bal_rec <- heart_recipe %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_smote(HeartDisease)


bal_wf <- workflow() %>% 
  add_recipe(bal_rec) %>% 
  add_model(bag_spec)

set.seed(123)
bal_results <- fit_resamples(
  bal_wf,
  resamples = heart_folds,
  metrics = heart_metrics,
  control = control_resamples(save_pred = TRUE))


collect_metrics(bal_results)

bal_results %>% 
  conf_mat_resampled()
```

```{r Fitting onto Test Data}
heart_final <- bal_wf %>% 
  last_fit(data_split)

collect_metrics(heart_final)

collect_predictions(heart_final) %>% 
  conf_mat(HeartDisease, .pred_class)
```

