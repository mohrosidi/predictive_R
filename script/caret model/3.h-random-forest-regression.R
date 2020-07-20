# 1. Import library ----

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(rsample)) install.packages("rsample")
if(!require(recipes)) install.packages("recipes")
if(!require(caret)) install.packages("caret")
if(!require(DataExplorer)) install.packages("DataExplorer")

## Helper packages
library(dplyr)      # for data wrangling
library(ggplot2)    # for awesome graphics
library(rsample)    # for creating validation splits
library(recipes)    # for feature engineering
library(DataExplorer) # for eda
library(modeldata)

## Modeling packages
library(caret)  
library(ranger)

# 2. Import dataset ----

data(ames)

# 3. Data Splitting ----

set.seed(123)
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)

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
  step_integer(matches("Qual|Cond|QC|Qu")) # %>%
  # step_center(all_numeric(), -all_outcomes()) %>%
  # step_scale(all_numeric(), -all_outcomes())  %>%
  # step_pca(all_numeric(), -all_outcomes())  %>%
# step_dummy(all_nominal(), -all_outcomes())

blueprint


# 6. Model ----

## Specify resampling plan
cv <- trainControl(
  # possible value: "boot", "boot632", "optimism_boot", "boot_all", "cv", 
  #                 "repeatedcv", "LOOCV", "LGOCV"
  method = "cv", 
  number = 10, 
  # repeats = 5,
  allowParallel = TRUE
)

## Construct grid of hyperparameter values
n_features <- length(setdiff(names(ames_train), "Sale_Price"))
hyper_grid <- expand.grid(
  mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
  min.node.size = c(1, 3, 5, 10),
  splitrule = c("variance", "extratrees", "maxstat", "beta" )
)

## Tune a knn model using grid search
start <- Sys.time()

rf_fit <- train(
  blueprint,
  data = ames_train,
  method = "ranger",
  trControl = cv,
  tuneGrid = hyper_grid)

end <- Sys.time()

end - start

rf_fit

## Visualize the model

### RMSE
ggplot(rf_fit)

### Variable importance
ggplot(varImp(rf_fit))

# 7. Predict -----
prediction <- predict(rf_fit, ames_test)

## RMSE
RMSE(prediction, ames_test$Sale_Price, na.rm = TRUE)
