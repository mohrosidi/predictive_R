# 1. Import library ----

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(rsample)) install.packages("rsample")
if(!require(recipes)) install.packages("recipes")
if(!require(tidymodels)) install.packages("tidymodels")
if(!require(DataExplorer)) install.packages("DataExplorer")
if(!require(kknn)) install.packages("kknn")
if(!require(vip)) install.packages("vip")

## Helper packages
library(dplyr)      # for data wrangling
library(ggplot2)    # for awesome graphics
library(rsample)    # for creating validation splits
library(recipes)    # for feature engineering
library(DataExplorer) # for eda
library(modeldata) # for dataset
library(vip) # fo variable importance plot


## Modeling packages
library(tidymodels)       
library(kknn) # for fitting knn model


# 2. Import dataset ----
data(attrition)
churn <- attrition

# 3. Data Splitting ----
set.seed(123)
split <- initial_split(churn, prop = 0.7, 
                       strata = "Attrition")
churn_train  <- training(split)
churn_test   <- testing(split)

# create CV object from training data
churn_cv <- vfold_cv(churn_train, v=10, repeats = 1)

# 4. EDA ----

## View basic description 
introduce(churn_train)

## Plot basic description for Attrition data
plot_intro(churn_train)

## View missing value distribution for Attrition data
plot_missing(churn_train)

## frequency distribution of all discrete variables
plot_bar(churn_train)

## View histogram of all continuous variables
plot_histogram(churn_train)

## View quantile-quantile plot of all continuous variables
plot_qq(churn_train)

## View overall correlation heatmap
plot_correlation(churn_train)

# 5. Feature & Target Engineering ----

# Feature Engineering Step
# 1. Filter out zero or near-zero variance features.
# 2. Perform imputation if required.
# 3. Normalize to resolve numeric feature skewness.
# 4. Standardize (center and scale) numeric features.
# 5. Perform dimension reduction (e.g., PCA) on numeric features.
# 6. One-hot or dummy encode categorical features.

## define feture engineering step
blueprint <- recipe(Attrition ~ ., data = churn_train) %>%
  step_nzv(all_nominal())  %>%
  
  # 2. imputation to missing value
  # step_medianimpute("<Num_Var_name>") %>% # median imputation
  # step_meanimpute("<Num_var_name>") %>% # mean imputation
  # step_modeimpute("<Cat_var_name>") %>% # mode imputation
  # step_bagimpute("<Var_name>") %>% # random forest imputation
  # step_knnimpute("<Var_name>") %>% # knn imputation
  
  # Label encoding for categorical variable with many classes 
  # step_integer("<Cat_var_name>") %>%
  
# 3. normalize to resolve numeric feature skewness
  step_center(all_numeric(), -all_outcomes()) %>%
  
  # 4. standardize (center and scale) numeric feature
  step_scale(all_numeric(), -all_outcomes()) %>%
  
  # 5. dimension reduction on numeric feature
  # step_pca(all_numeric(), -all_outcomes()) %>%
  
  # 6. one-hot or dummy encoding for categorical feature
  step_dummy(all_nominal(), -all_outcomes())

blueprint


# 6. Model ---

# Specify the model
knn_model <- 
  # specify that the model is a nearest_neigbor
  nearest_neighbor() %>%
  # specify that the `neighbors` parameter needs to be tuned
  set_args(neighbors = tune(),
           weight_func = tune()) %>%
  # select the engine/package that underlies the model
  set_engine("kknn") %>%
  # choose either the continuous regression or binary classification mode
  set_mode("classification")

# set the workflow
knn_workflow <- workflow() %>%
  # add the recipe
  add_recipe(blueprint) %>%
  # add the model
  add_model(knn_model)

# tune parameters

## specify which values want to try
knn_grid <- expand.grid(neighbors = seq(2,20, 1),
                        weight_func = c("rectangular", "triangular", "epanechnikov", 
                                        "biweight", "triweight", "cos", "inv", 
                                        "gaussian", "rank",  "optimal"))
## extract results
knn_tune_results <- knn_workflow %>%
  tune_grid(resamples = churn_cv, #CV object
            grid = knn_grid, # grid of values to try
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(accuracy, roc_auc) # metrics we care about
  )


## print results
knn_tune_results %>%
  collect_metrics() %>%
  arrange(-mean)

knn_best <- knn_tune_results %>%
  collect_metrics() %>%
  arrange(-mean) %>%
  slice(1)

## ROC
knn_auc <- 
  knn_tune_results %>% 
  collect_predictions() %>% 
  roc_curve(Attrition, .pred_Yes) %>% 
  mutate(model = "KNN Model")

autoplot(knn_auc)
autoplot(knn_tune_results)


# Finalize the workflow
param_final <- knn_tune_results %>%
  select_best(metric = "accuracy")
param_final

knn_workflow <- knn_workflow %>%
  finalize_workflow(param_final)


# 7. Evaluate the model on the test set

knn_fit <- knn_workflow %>%
  # fit on the training set and evaluate on test set
  last_fit(split)

knn_fit$.workflow

# test performance
test_performance <- knn_fit %>% collect_metrics()
test_performance

test_predictions <- knn_fit %>% collect_predictions()
test_predictions

# 8. Final Model ----

final_model <- fit(knn_workflow, churn)



# 9. Variable importance

knn_obj <- pull_workflow_fit(final_model)$fit


final_model %>% 
  pull_workflow_fit() %>% 
  vip(num_features = 20)
