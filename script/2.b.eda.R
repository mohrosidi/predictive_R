# 1. Import library ----

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(DataExplorer)) install.packages("DataExplorer")

library(tidyverse) # for data wrangling and visualization
library(DataExplorer) # for eda

# 2. Import dataset ----

data <- "<Import dataset>"

# 3. Data Splitting ----

set.seed(123)  # for reproducibility
split  <- initial_split(data, prop = "<train_prop>")
train  <- training(split_1)
test   <- testing(split_1)

# 4. EDA ----

## View basic description 
introduce(train)

## Plot basic description for AmesHousing data
plot_intro(train)

## View missing value distribution for AmesHousing data
plot_missing(train)

## frequency distribution of all discrete variables
plot_bar(train)

## View histogram of all continuous variables
plot_histogram(train)

## View quantile-quantile plot of all continuous variables
plot_qq(train)

## View overall correlation heatmap
plot_correlation(train)

## Scatterplot `price` with all other continuous features
plot_scatterplot(train, by = "<Cat_var_name>")

