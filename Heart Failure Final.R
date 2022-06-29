## Loading Libraries
```{r}
library(ggplot2)
library(skimr)
library(ranger)
library(tidyverse)
library(tidymodels)
library(GGally)
options(scipen=999)
library(doParallel)
library(themis)
```

## Loading Data

```{r}
heart <- read_csv("C:/Users/Dell/Desktop/Data Projects/Portfolio/ML Projects/Heart Failure/heart.csv", 
                  col_types = cols(Age = col_number(), 
                                   RestingBP = col_number(), Cholesterol = col_number(), 
                                   MaxHR = col_number(), Oldpeak = col_number()))
```

## EDA & Cleaning

### Skimming Data
```{r}
skim(heart)
```


### Verifying that the data is imbalanced
```{r}
heart %>% 
  count(HeartDisease)
```

### Variable Transformation

```{r}
heart$HeartDisease <- as.numeric(heart$HeartDisease)

heart <- heart %>% 
  mutate(HeartDisease = case_when(HeartDisease > 0.5 ~ "Yes", TRUE ~ "No"))


heart$HeartDisease <- as.factor(heart$HeartDisease)
heart$FastingBS <- as.character(heart$FastingBS)
```

### Numerical Pairs Plots
```{r}
heart %>% 
  select(HeartDisease, Age, RestingBP, Cholesterol, MaxHR, Oldpeak) %>% 
  ggpairs(columns = 2:6, aes(color = HeartDisease, alpha = 0.5))
```

### Categorical Pairs Plots
```{r}
heart %>% 
  select(HeartDisease, Sex, ChestPainType,RestingECG,ExerciseAngina,FastingBS,ST_Slope) %>% 
  pivot_longer(Sex:ST_Slope) %>% 
  ggplot(aes(y = value, fill = HeartDisease))+
  geom_bar(position = "fill")+
  facet_wrap(vars(name), scale = "free")+
  labs(x = NULL, y= NULL, fil = NULL)
```

## Modeling

### Splitting Data
```{r Splitting Data}
set.seed(123)
data_split <- initial_split(heart)
data_train <- training(data_split)
data_test <- testing(data_split)
```

### Generating Folds

```{r Folds}
heart_folds <- vfold_cv(data_train, v = 5, strata = HeartDisease)
heart_folds

heart_metrics <- metric_set(accuracy, sensitivity, specificity, recall)
```

### Recipe

```{r Recipe}
heart_recipe <- recipe(HeartDisease ~ ., data = data_train) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(), - all_outcomes()) %>% 
  step_zv(all_predictors())

heart_recipe
```

### Generating Model

```{r Bag Tree Model}

library(baguette)

bag_spec <-
  bag_tree(min_n = 10) %>% 
  set_engine("rpart", times = 25) %>% 
  set_mode("classification")

```

### Fitting Model

```{r Fitting Model to Data}
imb_wf <- workflow() %>% 
  add_recipe(heart_recipe) %>% 
  add_model(bag_spec)

var.imp.t <- fit(imb_wf, data = data_train)
```

### Accounting For Imbalance

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

### Balancing 

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

### Fitting onto Test Data

```{r Fitting onto Test Data}
heart_final <- bal_wf %>% 
  last_fit(data_split)

collect_metrics(heart_final)

collect_predictions(heart_final) %>% 
  conf_mat(HeartDisease, .pred_class)
```

## Variable Importance Table

```{r}
var.imp.t
```

## ROC Curve

```{r}
heart_final %>% 
  collect_predictions() %>% 
  group_by(id) %>% 
  roc_curve(HeartDisease, .pred_No) %>% 
  ggplot(aes(1 - specificity, sensitivity, color = id))+
  geom_abline(lty = 2, color = "gray90", size = 1.5)+
  geom_path(show.legend = FALSE, alpha = 0.6, size =1.2)+
  coord_equal()+theme_classic()
```

## Confusion Matrix

```{r}
collect_predictions(heart_final) %>% 
  conf_mat(HeartDisease, .pred_class) %>% 
  autoplot(cm, type = "heatmap")
```
