# 1. Import library ----

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(rsample)) install.packages("rsample")
if(!require(recipes)) install.packages("recipes")
if(!require(tidymodels)) install.packages("tidymodels")
if(!require(DataExplorer)) install.packages("DataExplorer")
if(!require(rpart)) install.packages("rpart")
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
library(parsnip)
library(workflows)
library(tune)
library(yardstick)
library(rpart) # for fitting decition tree model     

# 2. Import dataset ----

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
  step_integer(matches("Qual|Cond|QC|Qu"))  %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) # %>%
  # step_pca(all_numeric(), -all_outcomes())  %>%
  # step_dummy(all_nominal(), -all_outcomes())

blueprint


# 6. Model ----

# 6. Model ---

# Specify the model
dt_model <- 
  # specify that the model is a nearest_neigbor
  decision_tree() %>%
  # specify that the `neighbors` parameter needs to be tuned
  set_args(min_n = tune(),
           tree_depth = tune()) %>%
  # select the engine/package that underlies the model
  set_engine("rpart") %>%
  # choose either the continuous regression or binary classification mode
  set_mode("regression")

# set the workflow
dt_workflow <- workflow() %>%
  # add the recipe
  add_recipe(blueprint) %>%
  # add the model
  add_model(dt_model)

# tune parameters

## specify which values want to try
dt_grid <- expand.grid(min_n = seq(5, 20, 1),
                       tree_depth = seq(8, 30, 1))
## extract results
dt_tune_results <- dt_workflow %>%
  tune_grid(resamples = ames_cv, #CV object
            grid = dt_grid, # grid of values to try
            metrics = metric_set(rmse) # metrics we care about
  )


## print results
dt_tune_results %>%
  collect_metrics()

# Finalize the workflow
param_final <- dt_tune_results %>%
  select_best(metric = "rmse")
param_final

dt_workflow <- dt_workflow %>%
  finalize_workflow(param_final)


# 7. Evaluate the model on the test set

dt_fit <- dt_workflow %>%
  # fit on the training set and evaluate on test set
  last_fit(split)

dt_fit

# test performance
test_performance <- dt_fit %>% collect_metrics()
test_performance

test_predictions <- dt_fit %>% collect_predictions()
test_predictions

## Compare prediction vs actual
ggplot(test_predictions, aes(.pred, Sale_Price)) + geom_point() + geom_abline(slope = 1)


# 8. Final Model ----

final_model <- fit(dt_workflow, ames)
final_model


# 9. Variable importance

dt_obj <- pull_workflow_fit(final_model)$fit
barplot(dt_obj$variable.importance, horiz = TRUE)

final_model %>% 
  pull_workflow_fit() %>% 
  vip(num_features = 20)

