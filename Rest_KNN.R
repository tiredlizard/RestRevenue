library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(doParallel)
library(dials)

set.seed(1234567)

rest_train <- vroom('train.csv') %>%
  rename(city_group = `City Group`) %>%
  rename(open_date = `Open Date`) %>%
  mutate(open_year = year(as.Date(open_date, format = "%m/%d/%Y"))) %>%
  select(Id, open_year, City:revenue, -open_date)
rest_test <- vroom('test.csv') %>%
  rename(city_group = `City Group`) %>%
  rename(open_date = `Open Date`) %>%
  mutate(open_year = year(as.Date(open_date, format = "%m/%d/%Y"))) %>%
  select(Id, open_year, City:P37, -open_date)

rest_recipe <- recipe(revenue ~ ., data = rest_train) %>%
  step_rm(Id) %>%
  # step_mutate(City = as.factor(City), 
  #             city_group = as.factor(city_group), 
  #             Type = as.factor(Type)) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())

## Split data for CV
folds <- vfold_cv(bike_train_HW7, v = 5, repeats = 1)

## Create a control grid
untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a model


