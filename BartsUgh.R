library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(doParallel)
library(dials)

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
  step_mutate(City = as.factor(City), 
              city_group = as.factor(city_group), 
              Type = as.factor(Type)) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())


nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("regression") %>%
  set_engine("naivebayes") # install discrim library for naivebayes

nb_wf <- workflow() %>%
  add_recipe(rest_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace
nb_grid <- grid_regular(Laplace(), 
                        smoothness(),
                        levels = 10)

nb_folds <- vfold_cv(rest_train, v = 20, repeats = 1)

nb_CV <- nb_wf %>%
  tune_grid(resamples = nb_folds, 
            grid = nb_grid,
            metrics = metric_set(rmse))

bestTune_nb <- CV_results %>%
  select_best(metric = 'rmse')

nb_wf <-
  nb_wf %>%
  finalize_workflow(bestTune_nb) %>%
  fit(data = rest_train)


nb_preds <- predict(final_wf, new_data=rest_test)


submission <- bind_cols(rest_test$Id, preds) %>%
  rename(Id = ...1) %>%
  rename(Prediction = .pred)

write_csv(submission, 'Submission_nb1.csv')

