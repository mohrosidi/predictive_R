# 1. Import library ----

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(rsample)) install.packages("rsample")
if(!require(recipes)) install.packages("recipes")
if(!require(caret)) install.packages("caret")
if(!require(DataExplorer)) install.packages("DataExplorer")
if(!require(modeldata)) install.packages("modeldata")
if(!require(e1071)) install.packages("e1071")
if(!require(DMwR)) install.packages("DMwR")

## Helper packages
library(dplyr)      # for data wrangling
library(ggplot2)    # for awesome graphics
library(rsample)    # for creating validation splits
library(recipes)    # for feature engineering
library(DataExplorer) # for eda
library(modeldata) # for dataset
library(e1071)

## Modeling packages
library(caret)
library(ranger)

# 2. Import dataset ----
data("attrition")

churn <- attrition

# 3. Data Splitting ----
set.seed(123)
split <- initial_split(churn, prop = 0.7, 
                       strata = "Attrition")
churn_train  <- training(split)
churn_test   <- testing(split)

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
  step_nzv(all_nominal())  # %>%
  
  # 2. imputation to missing value
  # step_medianimpute("<Num_Var_name>") %>% # median imputation
  # step_meanimpute("<Num_var_name>") %>% # mean imputation
  # step_modeimpute("<Cat_var_name>") %>% # mode imputation
  # step_bagimpute("<Var_name>") %>% # random forest imputation
  # step_knnimpute("<Var_name>") %>% # knn imputation
  
  # Label encoding for categorical variable with many classes 
  # step_integer("<Cat_var_name>") %>%
  
  # 3. normalize to resolve numeric feature skewness
  # step_center(all_numeric(), -all_outcomes()) %>%
  
  # 4. standardize (center and scale) numeric feature
  # step_scale(all_numeric(), -all_outcomes()) # %>%

  # 5. dimension reduction on numeric feature
  # step_pca(all_numeric(), -all_outcomes()) %>%

  # 6. one-hot or dummy encoding for categorical feature
  # step_dummy(all_nominal(), -all_outcomes())

blueprint

# 6. Model ----

set.seed(123)

## Specify resampling plan
smotest <- list(name = "SMOTE with more neighbors!",
                func = function (x, y) {
                  library(DMwR)
                  dat <- if (is.data.frame(x)) x else as.data.frame(x)
                  dat$.y <- y
                  dat <- SMOTE(.y ~ ., data = dat, k = 10)
                  list(x = dat[, !grepl(".y", colnames(dat), fixed = TRUE)], 
                       y = dat$.y)
                },
                first = TRUE)

cv <- trainControl(
  # possible value: "boot", "boot632", "optimism_boot", "boot_all", "cv", 
  #                 "repeatedcv", "LOOCV", "LGOCV"
  method = "cv", 
  number = 10, 
  # repeats = 5,
  classProbs = TRUE,                 
  summaryFunction = twoClassSummary,
  sampling = smotest,
  allowParallel = TRUE
)

## Construct grid of hyperparameter values
n_features <- length(setdiff(names(churn_train), "Attrition"))
hyper_grid <- expand.grid(
  mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
  min.node.size = c(1, 3, 5, 10),
  splitrule = c("gini", "extratrees", "hellinger")
)

## Tune a knn model using grid search
start <- Sys.time()

rf_fit <- train(
  blueprint,
  data = churn_train,
  method = "ranger",
  trControl = cv,
  tuneGrid = hyper_grid,
  metric = "ROC")

end <- Sys.time()

end - start

rf_fit

## Visualize the model

### ROC
ggplot(rf_fit)

### Variable importance
ggplot(varImp(rf_fit))

# 7. Predict -----
prediction <- predict(rf_fit, churn_test)

## Confution matrix
caret::confusionMatrix(reference = churn_test$Attrition, data = prediction, mode='everything')
