# 04: Comprehensive Model Comparison and Overfitting Detection
# This script demonstrates systematic model comparison and overfitting analysis

# Load required libraries
library(tidymodels)
library(dplyr)
library(ggplot2)
library(purrr)

# Source utility functions
source("utils.R")

cat("=== Comprehensive Model Comparison and Overfitting Detection ===\n\n")

# Load mechanical strength dataset
strength_data <- readRDS("mechanical_strength_data.rds")
cat("Loaded mechanical strength dataset: ", nrow(strength_data), " samples\n\n")

# Step 1: Data Preparation
cat("Step 1: Data preparation and splitting\n")
set.seed(2023)
data_split <- initial_split(strength_data, prop = 0.75, strata = mechanical_strength_MPa)
train_data <- training(data_split)
test_data <- testing(data_split)

# Create CV folds
cv_folds <- vfold_cv(train_data, v = 5, strata = mechanical_strength_MPa)

cat("Training set: ", nrow(train_data), " samples\n")
cat("Test set: ", nrow(test_data), " samples\n")
cat("CV folds: 5-fold cross-validation\n\n")

# Step 2: Define Multiple Models with Different Complexity
cat("Step 2: Defining models with varying complexity\n")
cat("- Simple to complex models to demonstrate overfitting\n\n")

# Base preprocessing recipe
base_recipe <- recipe(mechanical_strength_MPa ~ ., data = train_data) %>%
  step_normalize(all_predictors())

# Model 1: Simple Linear Regression
linear_spec <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

linear_workflow <- workflow() %>%
  add_recipe(base_recipe) %>%
  add_model(linear_spec)

# Model 2: Polynomial Features (degree 2) - More complex
poly_recipe <- recipe(mechanical_strength_MPa ~ ., data = train_data) %>%
  step_poly(all_predictors(), degree = 2) %>%
  step_normalize(all_predictors())

poly_workflow <- workflow() %>%
  add_recipe(poly_recipe) %>%
  add_model(linear_spec)

# Model 3: High-degree Polynomial (degree 4) - Very complex, prone to overfitting
high_poly_recipe <- recipe(mechanical_strength_MPa ~ ., data = train_data) %>%
  step_poly(all_predictors(), degree = 4) %>%
  step_normalize(all_predictors())

high_poly_workflow <- workflow() %>%
  add_recipe(high_poly_recipe) %>%
  add_model(linear_spec)

# Model 4: Ridge Regression (regularized)
ridge_spec <- linear_reg(penalty = 0.1, mixture = 0) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

ridge_workflow <- workflow() %>%
  add_recipe(poly_recipe) %>%  # Use polynomial features with regularization
  add_model(ridge_spec)

# Model 5: Random Forest (ensemble method)
rf_spec <- rand_forest(trees = 100, mtry = 2) %>%
  set_engine("randomForest") %>%
  set_mode("regression")

rf_workflow <- workflow() %>%
  add_recipe(base_recipe) %>%
  add_model(rf_spec)

# Model 6: Overfitted Random Forest (too many trees, overfitting)
overfit_rf_spec <- rand_forest(trees = 1000, mtry = 1, min_n = 1) %>%
  set_engine("randomForest") %>%
  set_mode("regression")

overfit_rf_workflow <- workflow() %>%
  add_recipe(base_recipe) %>%
  add_model(overfit_rf_spec)

cat("Models defined:\n")
cat("1. Linear Regression (simple)\n")
cat("2. Polynomial degree 2 (moderate complexity)\n")
cat("3. Polynomial degree 4 (high complexity)\n")
cat("4. Ridge Regression (regularized)\n")
cat("5. Random Forest (ensemble)\n")
cat("6. Overfitted Random Forest (prone to overfitting)\n\n")

# Step 3: Evaluate All Models with Cross-Validation
cat("Step 3: Cross-validation evaluation of all models\n")

models <- list(
  "Linear Regression" = linear_workflow,
  "Polynomial Degree 2" = poly_workflow,
  "Polynomial Degree 4" = high_poly_workflow,
  "Ridge Regression" = ridge_workflow,
  "Random Forest" = rf_workflow,
  "Overfitted RF" = overfit_rf_workflow
)

# Function to evaluate a single model
evaluate_model <- function(workflow, name) {
  cat("Evaluating:", name, "\n")
  
  cv_results <- workflow %>%
    fit_resamples(
      resamples = cv_folds,
      metrics = metric_set(rsq, rmse, mae),
      control = control_resamples(save_pred = TRUE)
    )
  
  cv_metrics <- collect_metrics(cv_results)
  cv_metrics$model_name <- name
  
  return(cv_metrics)
}

# Evaluate all models
all_cv_results <- map2_dfr(models, names(models), evaluate_model)

cat("\nCross-validation completed for all models\n\n")

# Step 4: Training Set Performance (for overfitting comparison)
cat("Step 4: Evaluating training set performance\n")

evaluate_training <- function(workflow, name) {
  # Fit to training data
  fit <- workflow %>% fit(train_data)
  
  # Predict on training data
  train_pred <- fit %>%
    predict(train_data) %>%
    bind_cols(train_data)
  
  # Calculate metrics
  train_metrics <- train_pred %>%
    metrics(truth = mechanical_strength_MPa, estimate = .pred)
  
  train_metrics$model_name <- name
  train_metrics$dataset <- "training"
  
  return(train_metrics)
}

# Get training performance for all models
all_train_results <- map2_dfr(models, names(models), evaluate_training)

cat("Training set evaluation completed\n\n")

# Step 5: Overfitting Analysis
cat("Step 5: Overfitting analysis\n")

# Combine training and CV results for comparison
cv_rsq <- all_cv_results %>%
  filter(.metric == "rsq") %>%
  select(model_name, cv_rsq = mean, cv_se = std_err)

train_rsq <- all_train_results %>%
  filter(.metric == "rsq") %>%
  select(model_name, train_rsq = .estimate)

overfitting_analysis <- train_rsq %>%
  left_join(cv_rsq, by = "model_name") %>%
  mutate(
    overfitting_indicator = train_rsq - cv_rsq,
    overfitting_severity = case_when(
      overfitting_indicator < 0.05 ~ "None",
      overfitting_indicator < 0.15 ~ "Mild",
      overfitting_indicator < 0.25 ~ "Moderate", 
      TRUE ~ "Severe"
    )
  ) %>%
  arrange(desc(overfitting_indicator))

cat("Overfitting Analysis Results:\n")
print(overfitting_analysis)
cat("\n")

# Step 6: Model Ranking and Selection
cat("Step 6: Model ranking based on cross-validation performance\n")

model_ranking <- all_cv_results %>%
  filter(.metric == "rsq") %>%
  select(model_name, cv_rsq = mean, cv_se = std_err) %>%
  arrange(desc(cv_rsq))

cat("Model ranking by CV R-squared:\n")
print(model_ranking)
cat("\n")

# Step 7: Visualizations
cat("Step 7: Creating comprehensive visualizations\n")

# Overfitting comparison plot
p1 <- overfitting_analysis %>%
  select(model_name, train_rsq, cv_rsq) %>%
  pivot_longer(cols = c(train_rsq, cv_rsq), names_to = "dataset", values_to = "rsq") %>%
  ggplot(aes(x = reorder(model_name, rsq), y = rsq, fill = dataset)) +
  geom_col(position = "dodge", alpha = 0.7) +
  labs(
    title = "Training vs Cross-Validation Performance",
    subtitle = "Large gaps indicate overfitting",
    x = "Model",
    y = "R-squared",
    fill = "Dataset"
  ) +
  scale_fill_manual(values = c("train_rsq" = "lightblue", "cv_rsq" = "darkblue"),
                    labels = c("Cross-Validation", "Training")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p1)

# Overfitting indicator plot
p2 <- overfitting_analysis %>%
  ggplot(aes(x = reorder(model_name, overfitting_indicator), 
             y = overfitting_indicator, 
             fill = overfitting_severity)) +
  geom_col(alpha = 0.7) +
  geom_hline(yintercept = 0.1, linetype = "dashed", color = "red") +
  labs(
    title = "Overfitting Indicator by Model",
    subtitle = "Higher values indicate more overfitting",
    x = "Model",
    y = "Overfitting Indicator (Train R² - CV R²)",
    fill = "Severity"
  ) +
  scale_fill_manual(values = c("None" = "green", "Mild" = "yellow", 
                               "Moderate" = "orange", "Severe" = "red")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p2)

# Model performance comparison
p3 <- all_cv_results %>%
  filter(.metric %in% c("rsq", "rmse")) %>%
  ggplot(aes(x = reorder(model_name, ifelse(.metric == "rsq", -mean, mean)), 
             y = mean, fill = model_name)) +
  geom_col(alpha = 0.7) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), width = 0.2) +
  facet_wrap(~.metric, scales = "free_y") +
  labs(
    title = "Cross-Validation Performance Comparison",
    x = "Model",
    y = "Performance Metric",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

print(p3)

# Step 8: Final Model Selection and Test Set Evaluation
cat("Step 8: Final model selection and test evaluation\n")

# Select best model (highest CV R-squared with reasonable overfitting)
best_models <- overfitting_analysis %>%
  filter(overfitting_severity %in% c("None", "Mild")) %>%
  arrange(desc(cv_rsq))

if (nrow(best_models) > 0) {
  best_model_name <- best_models$model_name[1]
} else {
  # If all models overfit, choose the one with best CV performance
  best_model_name <- model_ranking$model_name[1]
}

cat("Selected best model: ", best_model_name, "\n")

# Evaluate on test set
final_workflow <- models[[best_model_name]]
final_fit <- final_workflow %>% fit(train_data)

test_predictions <- final_fit %>%
  predict(test_data) %>%
  bind_cols(test_data)

test_metrics <- test_predictions %>%
  metrics(truth = mechanical_strength_MPa, estimate = .pred)

cat("\nFinal test set performance:\n")
print(test_metrics)
cat("\n")

# Check for prediction problems
prediction_problems <- detect_prediction_problems(test_predictions$.pred)
cat("Prediction problems detected: ", prediction_problems, "\n\n")

# Step 9: Summary and Recommendations
cat("=== Model Comparison Summary ===\n")
cat("✓ Evaluated 6 models with different complexity levels\n")
cat("✓ Detected overfitting using train vs CV performance\n")
cat("✓ Ranked models by cross-validation performance\n")
cat("✓ Selected best model considering both performance and overfitting\n")
cat("✓ Final evaluation on independent test set\n\n")

cat("Key Findings:\n")
for (i in 1:nrow(overfitting_analysis)) {
  model <- overfitting_analysis$model_name[i]
  severity <- overfitting_analysis$overfitting_severity[i]
  indicator <- round(overfitting_analysis$overfitting_indicator[i], 3)
  cat("- ", model, ": ", severity, " overfitting (", indicator, ")\n")
}

cat("\nRecommendations:\n")
cat("1. Prefer models with good CV performance and low overfitting\n")
cat("2. Regularization (Ridge) can help reduce overfitting\n")
cat("3. Very complex models (high-degree polynomials) often overfit\n")
cat("4. Ensemble methods (Random Forest) can provide good balance\n")
cat("5. Always validate final model on independent test set\n")
