# 1. import library -------

library(tidyverse)
library(caret)
library(rsample)

# 2. import dataset ----
data <- "<Import dataset>"

# 3. simple random sampling ----

# Using base R
set.seed(123)  # for reproducibility
index_1 <- sample(1:nrow(data), round(nrow(data) * "<train_prop>"))
train_1 <- data[index_1, ]
test_1  <- data[-index_1, ]

# Using caret package
set.seed(123)  # for reproducibility
index_2 <- createDataPartition("<target_feature>", p = "<train_prop>", 
                               list = FALSE)
train_2 <- data[index_2, ]
test_2  <- data[-index_2, ]

# Using rsample package
set.seed(123)  # for reproducibility
split_1  <- initial_split(data, prop = "<train_prop>")
train_3  <- training(split_1)
test_3   <- testing(split_1)

# 4. Stratified Random Sampling ----

# orginal response distribution
table("<target_feature>") %>% prop.table()

# stratified sampling with the rsample package
set.seed(123)
split_strat  <- initial_split(data, prop = "<train_prop>", 
                              strata = "<target_feature>")
train_strat  <- training(split_strat)
test_strat   <- testing(split_strat)

# consistent response ratio between train & test
table(train_strat$Attrition) %>% prop.table()

table(test_strat$Attrition) %>% prop.table()

