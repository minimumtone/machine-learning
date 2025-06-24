# 02: Basic tidymodels Workflow with Train/Test Split
# This script demonstrates the fundamental tidymodels workflow for materials engineering

# Load required libraries
library(tidymodels)
library(dplyr)
library(ggplot2)

# Source utility functions
source("utils.R")

cat("=== Basic tidymodels Workflow ===\n\n")

# Load thermal conductivity dataset
thermal_data <- readRDS("thermal_conductivity_data.rds")
cat("Loaded thermal conductivity dataset: ", nrow(thermal_data), " samples\n\n")

# Step 1: Data Splitting (Critical for Proper Evaluation)
cat("Step 1: Splitting data into training and testing sets\n")
cat("- Training set: Used for model fitting and parameter estimation\n")
cat("- Test set: Used ONLY for final model evaluation (never during development)\n\n")

set.seed(123)
data_split <- initial_split(thermal_data, prop = 0.75, strata = thermal_conductivity_W_per_mK)
train_data <- training(data_split)
test_data <- testing(data_split)

cat("Training set size: ", nrow(train_data), " samples\n")
cat("Test set size: ", nrow(test_data), " samples\n")
cat("Split ratio: ", round(nrow(train_data) / nrow(thermal_data), 2), "\n\n")

# Step 2: Feature Engineering with recipes
cat("Step 2: Creating a preprocessing recipe\n")
cat("- Recipes define how to prepare data for modeling\n")
cat("- Can include scaling, transformations, feature creation\n\n")

thermal_recipe <- recipe(thermal_conductivity_W_per_mK ~ ., data = train_data) %>%
  step_normalize(all_predictors()) %>%  # Standardize features
  step_corr(all_predictors(), threshold = 0.9)  # Remove highly correlated features

cat("Recipe steps:\n")
cat("1. Normalize all predictor variables (mean=0, sd=1)\n") 
cat("2. Remove predictors with correlation > 0.9\n\n")

# Step 3: Model Specification
cat("Step 3: Specifying the model\n")
cat("- Define the type of model and computational engine\n")
cat("- Set mode (regression vs classification)\n\n")

linear_spec <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

cat("Model specification: Linear regression using 'lm' engine\n\n")

# Step 4: Create Workflow
cat("Step 4: Creating a workflow\n")
cat("- Workflows combine preprocessing and model specification\n")
cat("- Ensures consistent data processing across train/test\n\n")

thermal_workflow <- workflow() %>%
  add_recipe(thermal_recipe) %>%
  add_model(linear_spec)

cat("Workflow created: recipe + linear regression model\n\n")

# Step 5: Fit the Model
cat("Step 5: Fitting the model to training data\n")

thermal_fit <- thermal_workflow %>%
  fit(data = train_data)

cat("✓ Model fitted successfully\n\n")

# Step 6: Evaluate on Training Data (for comparison)
cat("Step 6: Evaluating model performance\n\n")

# Training set predictions
train_predictions <- thermal_fit %>%
  predict(train_data) %>%
  bind_cols(train_data)

train_metrics <- train_predictions %>%
  metrics(truth = thermal_conductivity_W_per_mK, estimate = .pred)

cat("Training Set Performance:\n")
print(train_metrics)
cat("\n")

# Test set predictions (final evaluation)
test_predictions <- thermal_fit %>%
  predict(test_data) %>%
  bind_cols(test_data)

test_metrics <- test_predictions %>%
  metrics(truth = thermal_conductivity_W_per_mK, estimate = .pred)

cat("Test Set Performance:\n")
print(test_metrics)
cat("\n")

# Step 7: Check for Overfitting
cat("Step 7: Overfitting Analysis\n")
overfitting_analysis <- calculate_overfitting_metrics(train_metrics, test_metrics)
print(overfitting_analysis)
cat("\n")

if (overfitting_analysis$overfitting_indicator > 0.1) {
  cat("⚠️  Warning: Potential overfitting detected!\n")
  cat("   Training R² is much higher than test R²\n\n")
} else {
  cat("✓ Good generalization: Similar performance on train and test sets\n\n")
}

# Step 8: Visualize Results
cat("Step 8: Visualizing model performance\n")

# Predicted vs Actual plot
p1 <- test_predictions %>%
  ggplot(aes(x = thermal_conductivity_W_per_mK, y = .pred)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Thermal Conductivity",
    subtitle = "Test Set Performance",
    x = "Actual Thermal Conductivity (W/m·K)",
    y = "Predicted Thermal Conductivity (W/m·K)"
  ) +
  theme_minimal()

print(p1)

# Residuals plot
p2 <- test_predictions %>%
  mutate(residuals = thermal_conductivity_W_per_mK - .pred) %>%
  ggplot(aes(x = .pred, y = residuals)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Residuals vs Predicted Values",
    subtitle = "Check for patterns in model errors",
    x = "Predicted Thermal Conductivity (W/m·K)",
    y = "Residuals"
  ) +
  theme_minimal()

print(p2)

# Step 9: Extract Model Coefficients
cat("Step 9: Model interpretation\n")

model_coefficients <- thermal_fit %>%
  extract_fit_parsnip() %>%
  tidy()

cat("Model coefficients:\n")
print(model_coefficients)
cat("\n")

# Step 10: Feature Importance (based on coefficient magnitude)
feature_importance <- model_coefficients %>%
  filter(term != "(Intercept)") %>%
  mutate(abs_estimate = abs(estimate)) %>%
  arrange(desc(abs_estimate))

cat("Feature importance (by coefficient magnitude):\n")
print(feature_importance)
cat("\n")

# Summary
cat("=== Workflow Summary ===\n")
cat("✓ Data properly split into train/test sets\n")
cat("✓ Preprocessing recipe created and applied\n")
cat("✓ Linear regression model fitted\n")
cat("✓ Performance evaluated on both train and test sets\n")
cat("✓ Overfitting analysis completed\n")
cat("✓ Results visualized and interpreted\n\n")

cat("Key Learning Points:\n")
cat("1. NEVER use test data during model development\n")
cat("2. Recipes ensure consistent preprocessing\n")
cat("3. Workflows combine preprocessing and modeling\n")
cat("4. Compare train vs test performance to detect overfitting\n")
cat("5. Visualize predictions to understand model behavior\n")
