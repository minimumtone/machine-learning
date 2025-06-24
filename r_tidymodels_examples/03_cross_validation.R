# 03: Cross-Validation and Model Tuning
# This script demonstrates k-fold cross-validation for robust model evaluation

# Load required libraries
library(tidymodels)
library(dplyr)
library(ggplot2)
library(purrr)

# Source utility functions
source("utils.R")

cat("=== Cross-Validation for Robust Model Evaluation ===\n\n")

# Load electrical conductivity dataset for this example
electrical_data <- readRDS("electrical_conductivity_data.rds")
cat("Loaded electrical conductivity dataset: ", nrow(electrical_data), " samples\n\n")

# Step 1: Data Splitting
cat("Step 1: Creating train/test split\n")
set.seed(456)
data_split <- initial_split(electrical_data, prop = 0.8, 
                           strata = electrical_conductivity_S_per_m)
train_data <- training(data_split)
test_data <- testing(data_split)

cat("Training set: ", nrow(train_data), " samples\n")
cat("Test set: ", nrow(test_data), " samples\n\n")

# Step 2: Create Cross-Validation Folds
cat("Step 2: Creating cross-validation folds\n")
cat("- k-fold CV splits training data into k subsets\n")
cat("- Each fold serves as validation set once\n")
cat("- Provides more robust performance estimates\n\n")

set.seed(789)
cv_folds <- vfold_cv(train_data, v = 5, strata = electrical_conductivity_S_per_m)

cat("Created 5-fold cross-validation:\n")
cat("- Each fold: ~", round(nrow(train_data) * 0.8), " training, ~", 
    round(nrow(train_data) * 0.2), " validation samples\n\n")

# Step 3: Define Multiple Models for Comparison
cat("Step 3: Defining multiple models for comparison\n\n")

# Linear regression
linear_spec <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# Ridge regression with tuning parameter
ridge_spec <- linear_reg(penalty = tune(), mixture = 0) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Random forest
rf_spec <- rand_forest(trees = 500, mtry = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression")

cat("Models defined:\n")
cat("1. Linear regression (no tuning)\n")
cat("2. Ridge regression (penalty parameter tuned)\n") 
cat("3. Random forest (mtry parameter tuned)\n\n")

# Step 4: Create Preprocessing Recipe
cat("Step 4: Creating preprocessing recipe\n")

electrical_recipe <- recipe(electrical_conductivity_S_per_m ~ ., data = train_data) %>%
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors())  # Remove zero-variance predictors

cat("Recipe steps: normalize predictors, remove zero-variance features\n\n")

# Step 5: Cross-Validation for Linear Regression
cat("Step 5: Cross-validation for linear regression\n")

linear_workflow <- workflow() %>%
  add_recipe(electrical_recipe) %>%
  add_model(linear_spec)

linear_cv_results <- linear_workflow %>%
  fit_resamples(
    resamples = cv_folds,
    metrics = metric_set(rsq, rmse, mae),
    control = control_resamples(save_pred = TRUE)
  )

linear_cv_metrics <- collect_metrics(linear_cv_results)
cat("Linear regression CV results:\n")
print(linear_cv_metrics)
cat("\n")

# Step 6: Cross-Validation with Hyperparameter Tuning (Ridge)
cat("Step 6: Ridge regression with hyperparameter tuning\n")

ridge_workflow <- workflow() %>%
  add_recipe(electrical_recipe) %>%
  add_model(ridge_spec)

# Define penalty parameter grid
penalty_grid <- grid_regular(penalty(range = c(-3, 1)), levels = 10)

ridge_cv_results <- ridge_workflow %>%
  tune_grid(
    resamples = cv_folds,
    grid = penalty_grid,
    metrics = metric_set(rsq, rmse, mae)
  )

# Show best parameters
best_ridge <- select_best(ridge_cv_results, metric = "rmse")
cat("Best ridge penalty parameter:\n")
print(best_ridge)
cat("\n")

# Finalize ridge workflow with best parameters
final_ridge_workflow <- ridge_workflow %>%
  finalize_workflow(best_ridge)

ridge_cv_metrics <- final_ridge_workflow %>%
  fit_resamples(
    resamples = cv_folds,
    metrics = metric_set(rsq, rmse, mae)
  ) %>%
  collect_metrics()

cat("Ridge regression CV results (best parameters):\n")
print(ridge_cv_metrics)
cat("\n")

# Step 7: Random Forest with Tuning
cat("Step 7: Random forest with hyperparameter tuning\n")

rf_workflow <- workflow() %>%
  add_recipe(electrical_recipe) %>%
  add_model(rf_spec)

# Define mtry parameter grid (number of features to consider at each split)
rf_grid <- grid_regular(mtry(range = c(1, 3)), levels = 3)

rf_cv_results <- rf_workflow %>%
  tune_grid(
    resamples = cv_folds,
    grid = rf_grid,
    metrics = metric_set(rsq, rmse, mae)
  )

best_rf <- select_best(rf_cv_results, metric = "rmse")
cat("Best random forest mtry parameter:\n")
print(best_rf)
cat("\n")

final_rf_workflow <- rf_workflow %>%
  finalize_workflow(best_rf)

rf_cv_metrics <- final_rf_workflow %>%
  fit_resamples(
    resamples = cv_folds,
    metrics = metric_set(rsq, rmse, mae)
  ) %>%
  collect_metrics()

cat("Random forest CV results (best parameters):\n")
print(rf_cv_metrics)
cat("\n")

# Step 8: Compare Models
cat("Step 8: Model comparison based on cross-validation\n")

# Combine results
model_comparison <- bind_rows(
  linear_cv_metrics %>% mutate(model = "Linear Regression"),
  ridge_cv_metrics %>% mutate(model = "Ridge Regression"),
  rf_cv_metrics %>% mutate(model = "Random Forest")
)

# Focus on R-squared for comparison
rsq_comparison <- model_comparison %>%
  filter(.metric == "rsq") %>%
  select(model, mean, std_err) %>%
  arrange(desc(mean))

cat("Model ranking by R-squared (cross-validation):\n")
print(rsq_comparison)
cat("\n")

# RMSE comparison
rmse_comparison <- model_comparison %>%
  filter(.metric == "rmse") %>%
  select(model, mean, std_err) %>%
  arrange(mean)

cat("Model ranking by RMSE (cross-validation):\n")
print(rmse_comparison)
cat("\n")

# Step 9: Visualize Cross-Validation Results
cat("Step 9: Visualizing cross-validation results\n")

# R-squared comparison plot
p1 <- model_comparison %>%
  filter(.metric == "rsq") %>%
  ggplot(aes(x = reorder(model, mean), y = mean, fill = model)) +
  geom_col(alpha = 0.7) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), 
                width = 0.2) +
  labs(
    title = "Model Comparison: R-squared (Cross-Validation)",
    subtitle = "Higher is better",
    x = "Model Type",
    y = "R-squared",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p1)

# RMSE comparison plot
p2 <- model_comparison %>%
  filter(.metric == "rmse") %>%
  ggplot(aes(x = reorder(model, -mean), y = mean, fill = model)) +
  geom_col(alpha = 0.7) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), 
                width = 0.2) +
  labs(
    title = "Model Comparison: RMSE (Cross-Validation)",
    subtitle = "Lower is better",
    x = "Model Type", 
    y = "RMSE",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p2)

# Step 10: Final Model Selection and Test Set Evaluation
cat("Step 10: Final model evaluation on test set\n")

# Select best model based on CV performance
best_model_name <- rsq_comparison$model[1]
cat("Best model based on CV R-squared: ", best_model_name, "\n\n")

# Fit best model to full training set and evaluate on test set
if (best_model_name == "Linear Regression") {
  final_workflow <- linear_workflow
} else if (best_model_name == "Ridge Regression") {
  final_workflow <- final_ridge_workflow
} else {
  final_workflow <- final_rf_workflow
}

# Fit to full training set
final_fit <- final_workflow %>%
  fit(train_data)

# Evaluate on test set
test_predictions <- final_fit %>%
  predict(test_data) %>%
  bind_cols(test_data)

test_metrics <- test_predictions %>%
  metrics(truth = electrical_conductivity_S_per_m, estimate = .pred)

cat("Final test set performance (", best_model_name, "):\n")
print(test_metrics)
cat("\n")

# Step 11: Learning Curve Analysis
cat("Step 11: Learning curve analysis\n")
cat("- Shows how performance changes with training set size\n")
cat("- Helps diagnose overfitting vs underfitting\n\n")

# Create learning curve data
learning_curve_data <- map_dfr(seq(0.2, 1.0, by = 0.2), function(prop) {
  # Sample training data
  sample_data <- train_data %>% slice_sample(prop = prop)
  
  # Fit model
  temp_fit <- final_workflow %>% fit(sample_data)
  
  # Evaluate on training subset
  train_pred <- temp_fit %>% predict(sample_data) %>% bind_cols(sample_data)
  train_rsq <- train_pred %>% rsq(truth = electrical_conductivity_S_per_m, estimate = .pred) %>% pull(.estimate)
  
  # Evaluate on validation set (fixed)
  val_pred <- temp_fit %>% predict(test_data) %>% bind_cols(test_data)
  val_rsq <- val_pred %>% rsq(truth = electrical_conductivity_S_per_m, estimate = .pred) %>% pull(.estimate)
  
  tibble(
    training_size = nrow(sample_data),
    train_rsq = train_rsq,
    validation_rsq = val_rsq
  )
})

# Plot learning curve
p3 <- learning_curve_data %>%
  pivot_longer(cols = c(train_rsq, validation_rsq), 
               names_to = "set", values_to = "rsq") %>%
  ggplot(aes(x = training_size, y = rsq, color = set)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  labs(
    title = "Learning Curve Analysis",
    subtitle = "Performance vs Training Set Size",
    x = "Training Set Size",
    y = "R-squared",
    color = "Dataset"
  ) +
  scale_color_manual(values = c("train_rsq" = "blue", "validation_rsq" = "red"),
                     labels = c("Training", "Validation")) +
  theme_minimal()

print(p3)

# Summary
cat("\n=== Cross-Validation Summary ===\n")
cat("✓ 5-fold cross-validation implemented\n")
cat("✓ Multiple models compared fairly\n")
cat("✓ Hyperparameters tuned using CV\n")
cat("✓ Best model selected based on CV performance\n")
cat("✓ Final evaluation on held-out test set\n")
cat("✓ Learning curve analysis completed\n\n")

cat("Key Learning Points:\n")
cat("1. Cross-validation provides robust performance estimates\n")
cat("2. Hyperparameter tuning should use CV, not test set\n")
cat("3. Model selection based on CV performance\n")
cat("4. Test set used only for final, unbiased evaluation\n")
cat("5. Learning curves help diagnose overfitting/underfitting\n")
