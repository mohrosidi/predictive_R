# 1. import library -------

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(rsample)) install.packages("rsample")
if(!require(recipes)) install.packages("recipes")
if(!require(caret)) install.packages("caret")

library(tidyverse)
library(caret)
library(rsample)
library(recipes)

# 2. import dataset ----
data <- "<Change this>"

## statified sampling
set.seed(123)
split <- initial_split(data, prop = "<training proportion>", 
                       strata = "<Var_name>")
train  <- training(split)
test   <- testing(split)


# 3. feature engineering ----

# Feature Engineering Step
# 1. Filter out zero or near-zero variance features.
# 2. Perform imputation if required.
# 3. Normalize to resolve numeric feature skewness.
# 4. Standardize (center and scale) numeric features.
# 5. Perform dimension reduction (e.g., PCA) on numeric features.
# 6. One-hot or dummy encode categorical features.

blueprint <- recipe("<Model_formula>", data = train) %>%
  # 1. remove variabel with near zero values 
  step_nzv(all_nominal())  %>%
  # step_zv(all_nominal()) %>%
  
  # 2. imputation to missing value
  # step_medianimpute("<Num_Var_name>") %>% # median imputation
  # step_meanimpute("<Num_var_name>") %>% # mean imputation
  # step_modeimpute("<Cat_var_name>") %>% # mode imputation
  # step_bagimpute("<Var_name>") %>% # random forest imputation
  # step_knnimpute("<Var_name>") %>% # knn imputation
  
  # Label encoding for categorical variable with many classes 
  step_integer("<Cat_var_name>") %>%
  
  # 3. normalize to resolve numeric feature skewness
  step_center(all_numeric(), -all_outcomes()) %>%
  
  # 4. standardize (center and scale) numeric feature
  step_scale(all_numeric(), -all_outcomes()) %>%
  
  # 5. dimension reduction on numeric feature
  # step_pca(all_numeric(), -all_outcomes()) %>%
  
  # 6. one-hot or dummy encoding for categorical feature
  # step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)

blueprint

# 4. train the training data ----

prepare <- prep(blueprint, training = train)

prepare

# 5. apply the training data to blueprint

baked_train <- bake(prepare, new_data = train)
baked_test <- bake(prepare, new_data = test)
baked_train

