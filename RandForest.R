library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(doParallel)
library(dials)

set.seed(1228)

rest_train <- vroom('train.csv') %>%
  rename(city_group = `City Group`) %>%
  rename(open_date = `Open Date`) %>%
  mutate(open_date = mdy(open_date))
rest_test <- vroom('test.csv') %>%
  rename(city_group = `City Group`) %>%
  rename(open_date = `Open Date`) %>%
  mutate(open_date = mdy(open_date))



rest_recipe <- recipe(revenue ~ ., data = rest_train) %>%
  step_rm(Id) %>%
  step_date(open_date, features = c("year", "month", "doy", "decimal")) %>%  # Extracts year, month, day of year
  step_rm(open_date) %>%
  step_other(all_nominal_predictors(), threshold = .05) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())


prepped <- prep(rest_recipe)
baked <- bake(prepped, new_data = rest_train)

# NO TUNING GRID
# forest_mod <- rand_forest(mtry = 5,
#                           min_n = 2,
#                           trees = 300) %>%
#   set_engine("ranger") %>%
#   set_mode("regression")
# 
# forest_wf <- workflow() %>%
#   add_recipe(rest_recipe) %>%
#   add_model(forest_mod) %>%
#   fit(data = rest_train)

forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = tune()) %>%
  set_engine("ranger") %>%
  set_mode("regression")

forest_wf <- workflow() %>%
  add_recipe(rest_recipe) %>%
  add_model(forest_mod)

## BEST TUNE ORIG: mtry = 8, min_n = 14, trees = 667
## BEST TUNE SIMPLE: mtry = 4, min_n = 2, trees = 1333
forest_grid <- grid_regular(mtry(range = c(4, 42)), 
                                min_n(),
                                trees(), 
                                levels = 4) 

forest_folds <- vfold_cv(rest_train, v = 5, repeats = 2) 

forest_CV <- forest_wf %>%
  tune_grid(resamples = forest_folds,
            grid = forest_grid,
            metrics = metric_set(rmse))

bestTune_cv <- forest_CV %>%
  select_best(metric = 'rmse')

forest_wf <-
  forest_wf %>%
  finalize_workflow(bestTune_cv) %>%
  fit(data = rest_train)

preds <- predict(forest_wf, new_data = rest_test)

submission <- bind_cols(rest_test$Id, preds) %>%
  rename(Id = ...1) %>%
  rename(Prediction = .pred)

write_csv(submission, 'Submission_forest6.csv')

