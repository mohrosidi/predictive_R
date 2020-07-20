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
data(ames)
ames <- ames

# 3. Data Splitting ----

set.seed(123)
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)

# create CV object from training data
ames_cv <- vfold_cv(ames_train, v=10, repeats = 1)

# 4. EDA ----

## View basic description 
introduce(ames_train)

## Plot basic description for AmesHousing data
plot_intro(ames_train)

## View missing value distribution for AmesHousing data
plot_missing(ames_train)

## frequency distribution of all discrete variables
plot_bar(ames_train)

## View histogram of all continuous variables
plot_histogram(ames_train)

## View quantile-quantile plot of all continuous variables
plot_qq(ames_train)

## View overall correlation heatmap
plot_correlation(ames_train)

## Scatterplot `price` with all other continuous features
plot_scatterplot(ames_train, by = "Sale_Price")

# 5. Feature & Target Engineering ----

# Feature Engineering Step
# 1. Filter out zero or near-zero variance features.
# 2. Perform imputation if required.
# 3. Normalize to resolve numeric feature skewness.
# 4. Standardize (center and scale) numeric features.
# 5. Perform dimension reduction (e.g., PCA) on numeric features.
# 6. One-hot or dummy encode categorical features.

## define feture engineering step
blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_nzv(all_nominal())  %>%
  step_integer(matches("Qual|Cond|QC|Qu")) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_pca(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)

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
  set_mode("regression")

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
  tune_grid(resamples = ames_cv, #CV object
            grid = knn_grid, # grid of values to try
            metrics = metric_set(rmse, mape) # metrics we care about
  )


## print results
knn_tune_results %>%
  collect_metrics()

# Finalize the workflow
param_final <- knn_tune_results %>%
  select_best(metric = "rmse")
param_final

knn_workflow <- knn_workflow %>%
  finalize_workflow(param_final)


# 7. Evaluate the model on the test set

knn_fit <- knn_workflow %>%
  # fit on the training set and evaluate on test set
  last_fit(split)

knn_fit

# test performance
test_performance <- knn_fit %>% collect_metrics()
test_performance

test_predictions <- knn_fit %>% collect_predictions()
test_predictions

## Compare prediction vs actual
ggplot(test_predictions, aes(.pred, Sale_Price)) + geom_point() + geom_abline(slope = 1)


# 8. Final Model ----

final_model <- fit(knn_workflow, ames)
final_model


# 9. Variable importance

knn_obj <- pull_workflow_fit(final_model)$fit
knn_obj

final_model %>% 
  pull_workflow_fit() %>% 
  vip(num_features = 20)

