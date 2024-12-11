library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(doParallel)
library(dials)


# Data cleaning & setup ---------------------------------------------------

rest_train <- vroom('train.csv') %>%
  rename(city_group = `City Group`) %>%
  rename(open_date = `Open Date`) %>%
  mutate(open_year = year(as.Date(open_date, format = "%m/%d/%Y"))) %>%
  mutate(years_opened = 2024-as.numeric(open_year)) %>%
  select(Id, open_year, years_opened, City:revenue, -open_date)
rest_test <- vroom('test.csv') %>%
  rename(city_group = `City Group`) %>%
  rename(open_date = `Open Date`) %>%
  mutate(open_year = year(as.Date(open_date, format = "%m/%d/%Y"))) %>%
  mutate(years_opened = 2024-as.numeric(open_year)) %>%
  select(Id, open_year, years_opened, City:P37, -open_date)


# Switch City to Region for kicks and giggles

# Combine the training and test datasets
combined_data <- bind_rows(
  rest_train %>% mutate(set = "train"),
  rest_test %>% mutate(set = "test")
)

# Add the region column for the combined dataset
combined_data <- combined_data %>%
  mutate(region = case_when(
    City %in% c("İstanbul", "Edirne", "Kocaeli", "Bursa", "Tekirdağ", "Sakarya", 
                "Kırklareli", "Yalova", "Bilecik", "Kırşehir", "Çorum", 
                "Çanakkale", "Bolu") ~ "Marmara",
    City %in% c("İzmir", "Manisa", "Aydın", "Muğla", "Uşak", "Denizli", "Kütahya") ~ "Aegean",
    City %in% c("Trabzon", "Giresun", "Ordu", "Artvin", "Rize", "Zonguldak", "Samsun", 
                "Amasya") ~ "Black Sea",
    City %in% c("Ankara", "Konya", "Eskişehir", "Kayseri", "Nevşehir", "Kırıkkale", 
                "Bolu", "Sivas", "Çankırı", "Kırşehir", "Aksaray") ~ "Central Anatolia",
    City %in% c("Erzurum", "Erzincan", "Kars", "Malatya", "Van", "Diyarbakır") ~ "Eastern Anatolia",
    City %in% c("Gaziantep", "Şanlıurfa", "Adana", "Mardin", "Kahramanmaraş", "Batman", 
                "Siirt", "Osmaniye", "Hatay") ~ "Southeastern Anatolia",
    City %in% c("Antalya", "Mersin", "İsparta", "Osmaniye", "Aydın") ~ "Mediterranean",
    City == "Tanımsız" ~ "Undefined",
    TRUE ~ "Unknown"  # For any other unmatched cases
  ))

# Separate the combined data back into train and test sets
rest_train_2 <- combined_data %>% filter(set == "train") %>% select(-set)
rest_test_2 <- combined_data %>% filter(set == "test") %>% select(-set)

# Check if the regions were correctly assigned
table(rest_train_2$region)
table(rest_test_2$region)


# Model & recipe ----------------------------------------------------------

# recipe
rest_recipe <- recipe(revenue ~ ., data = rest_train_2) %>%
  step_lencode_mixed(Type, outcome = vars(revenue)) %>%
  step_lencode_mixed(region, outcome = vars(revenue)) %>%
  step_dummy(city_group, one_hot = TRUE)


# model
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees= tune()) %>% # or 1000
  set_engine("ranger") %>%
  set_mode("regression")

forest_wf <- workflow() %>%
  add_recipe(rest_recipe) %>%
  add_model(forest_mod)


# Define the parameters
mtry_param <- finalize(mtry(), rest_train_2)

#tuning grid
forest_grid <- grid_regular(mtry(range = c(2, 20)),
                            min_n(),
                            trees(),
                            levels = 4)



# Cv ----------------------------------------------------------------------

forest_folds <- vfold_cv(rest_train_2, v = 4, repeats = 1)

CV_results <- forest_wf %>%
  tune_grid(resamples = forest_folds,
            grid = forest_grid,
            metrics = metric_set(rmse))

bestTune_cf <- CV_results_cf %>%
  select_best(metric = 'rmse')


setdiff(unique(rest_test_2$open_year), unique(rest_train$City))

unique(rest_test$City)

# workflow




# preds