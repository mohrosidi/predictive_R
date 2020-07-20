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

## Modeling packages
library(caret)       # for fitting KNN models

# 2. Import dataset ----

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
  step_dummy(all_nominal(), -all_outcomes(), one_hot = FALSE)

blueprint

# 6. Model ----

## Specify resampling plan
cv <- trainControl(
  # possible value: "boot", "boot632", "optimism_boot", "boot_all", "cv", 
  #                 "repeatedcv", "LOOCV", "LGOCV"
  method = "cv", 
  number = 10, 
  # repeats = 5,
  classProbs = TRUE,                 
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

## Construct grid of hyperparameter values
hyper_grid <- expand.grid(
  # kernel density estimate for continous variables vs gaussian density estimate
  usekernel = c(TRUE, FALSE),
  # incorporate the Laplace smoother.
  fL = 0:5,
  # djust the bandwidth of the kernel density (larger numbers mean more flexible density estimate)
  adjust = seq(0, 5, by = 1)
)

## Tune a knn model using grid search
start <- Sys.time()

nb_fit <- train(
  blueprint, 
  data = churn_train, 
  method = "nb", 
  trControl = cv, 
  tuneGrid = hyper_grid,
  metric = "Accuracy"
)

end <- Sys.time()

end - start

nb_fit

## Visualize the model

### ROC
ggplot(nb_fit)

### Variable importance
ggplot(varImp(nb_fit))

# 7. Predict -----
prediction <- predict(nb_fit, churn_test)

## Confution matrix
caret::confusionMatrix(reference = churn_test$Attrition, data = prediction, mode='everything')


