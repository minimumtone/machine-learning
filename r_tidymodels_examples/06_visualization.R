# 06: Comprehensive Visualization and Model Diagnostics
# This script focuses on creating informative visualizations for ML model evaluation

# Load required libraries
library(tidymodels)
library(dplyr)
library(ggplot2)
library(purrr)
library(gridExtra)
library(corrplot)

# Source utility functions
source("utils.R")

cat("=== Comprehensive Visualization and Model Diagnostics ===\n\n")

# Load all datasets for comprehensive visualization
thermal_data <- readRDS("thermal_conductivity_data.rds")
electrical_data <- readRDS("electrical_conductivity_data.rds")
strength_data <- readRDS("mechanical_strength_data.rds")

cat("Loaded all three materials datasets\n\n")

# Step 1: Exploratory Data Analysis Visualizations
cat("Step 1: Exploratory Data Analysis Visualizations\n")

# Function to create correlation heatmap
create_correlation_plot <- function(data, title) {
  cor_matrix <- cor(data)
  
  # Convert to long format for ggplot
  cor_long <- cor_matrix %>%
    as.data.frame() %>%
    rownames_to_column("var1") %>%
    pivot_longer(-var1, names_to = "var2", values_to = "correlation")
  
  ggplot(cor_long, aes(x = var1, y = var2, fill = correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                         midpoint = 0, limit = c(-1, 1)) +
    labs(title = paste("Correlation Matrix:", title),
         x = "", y = "", fill = "Correlation") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Create correlation plots for all datasets
p1 <- create_correlation_plot(thermal_data, "Thermal Conductivity")
p2 <- create_correlation_plot(electrical_data, "Electrical Conductivity") 
p3 <- create_correlation_plot(strength_data, "Mechanical Strength")

print(p1)
print(p2)
print(p3)

# Step 2: Distribution Analysis
cat("Step 2: Distribution analysis of target variables\n")

# Combine all target variables for comparison
target_distributions <- bind_rows(
  thermal_data %>% 
    select(value = thermal_conductivity_W_per_mK) %>% 
    mutate(property = "Thermal Conductivity (W/m·K)"),
  electrical_data %>% 
    select(value = electrical_conductivity_S_per_m) %>% 
    mutate(property = "Electrical Conductivity (S/m)"),
  strength_data %>% 
    select(value = mechanical_strength_MPa) %>% 
    mutate(property = "Mechanical Strength (MPa)")
)

# Distribution comparison plot
p4 <- target_distributions %>%
  ggplot(aes(x = value, fill = property)) +
  geom_histogram(alpha = 0.7, bins = 30) +
  facet_wrap(~property, scales = "free") +
  labs(
    title = "Distribution of Material Properties",
    x = "Property Value",
    y = "Frequency",
    fill = "Property"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

print(p4)

# Step 3: Model Performance Visualization Framework
cat("Step 3: Setting up model performance visualization framework\n")

# Function to create comprehensive model evaluation plots
create_model_diagnostics <- function(predictions, actual_col, pred_col, title) {
  
  # Calculate residuals
  predictions <- predictions %>%
    mutate(
      residuals = !!sym(actual_col) - !!sym(pred_col),
      abs_residuals = abs(residuals)
    )
  
  # 1. Predicted vs Actual
  p1 <- predictions %>%
    ggplot(aes(x = !!sym(actual_col), y = !!sym(pred_col))) +
    geom_point(alpha = 0.6, color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(
      title = paste(title, "- Predicted vs Actual"),
      x = "Actual Values",
      y = "Predicted Values"
    ) +
    theme_minimal()
  
  # 2. Residuals vs Predicted
  p2 <- predictions %>%
    ggplot(aes(x = !!sym(pred_col), y = residuals)) +
    geom_point(alpha = 0.6, color = "darkgreen") +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    geom_smooth(method = "loess", se = TRUE, color = "blue") +
    labs(
      title = paste(title, "- Residuals vs Predicted"),
      x = "Predicted Values",
      y = "Residuals"
    ) +
    theme_minimal()
  
  # 3. Residuals distribution
  p3 <- predictions %>%
    ggplot(aes(x = residuals)) +
    geom_histogram(bins = 30, alpha = 0.7, fill = "lightblue", color = "black") +
    geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
    labs(
      title = paste(title, "- Residuals Distribution"),
      x = "Residuals",
      y = "Frequency"
    ) +
    theme_minimal()
  
  # 4. Q-Q plot for residuals
  p4 <- predictions %>%
    ggplot(aes(sample = residuals)) +
    stat_qq() +
    stat_qq_line(color = "red") +
    labs(
      title = paste(title, "- Q-Q Plot of Residuals"),
      x = "Theoretical Quantiles",
      y = "Sample Quantiles"
    ) +
    theme_minimal()
  
  return(list(pred_vs_actual = p1, residuals_vs_pred = p2, 
              residuals_dist = p3, qq_plot = p4))
}

# Step 4: Demonstrate Model Diagnostics with Example
cat("Step 4: Demonstrating model diagnostics with thermal conductivity example\n")

# Quick model fitting for demonstration
set.seed(123)
thermal_split <- initial_split(thermal_data, prop = 0.8)
thermal_train <- training(thermal_split)
thermal_test <- testing(thermal_split)

# Simple linear model for demonstration
thermal_recipe <- recipe(thermal_conductivity_W_per_mK ~ ., data = thermal_train) %>%
  step_normalize(all_predictors())

linear_spec <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

thermal_workflow <- workflow() %>%
  add_recipe(thermal_recipe) %>%
  add_model(linear_spec)

thermal_fit <- thermal_workflow %>% fit(thermal_train)

# Generate predictions
thermal_predictions <- thermal_fit %>%
  predict(thermal_test) %>%
  bind_cols(thermal_test)

# Create diagnostic plots
diagnostic_plots <- create_model_diagnostics(
  thermal_predictions, 
  "thermal_conductivity_W_per_mK", 
  ".pred",
  "Thermal Conductivity Model"
)

# Display diagnostic plots
print(diagnostic_plots$pred_vs_actual)
print(diagnostic_plots$residuals_vs_pred)
print(diagnostic_plots$residuals_dist)
print(diagnostic_plots$qq_plot)

# Step 5: Cross-Validation Results Visualization
cat("Step 5: Cross-validation results visualization\n")

# Perform CV for multiple models
cv_folds <- vfold_cv(thermal_train, v = 5)

# Define multiple models
models <- list(
  "Linear" = linear_reg() %>% set_engine("lm"),
  "Ridge" = linear_reg(penalty = 0.1, mixture = 0) %>% set_engine("glmnet"),
  "Random Forest" = rand_forest(trees = 100) %>% set_engine("randomForest")
)

# Evaluate all models
cv_results <- map_dfr(names(models), function(model_name) {
  
  model_workflow <- workflow() %>%
    add_recipe(thermal_recipe) %>%
    add_model(models[[model_name]])
  
  cv_res <- model_workflow %>%
    fit_resamples(
      resamples = cv_folds,
      metrics = metric_set(rsq, rmse, mae)
    )
  
  collect_metrics(cv_res) %>%
    mutate(model = model_name)
})

# CV results visualization
p5 <- cv_results %>%
  filter(.metric == "rsq") %>%
  ggplot(aes(x = reorder(model, mean), y = mean, fill = model)) +
  geom_col(alpha = 0.7) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), width = 0.2) +
  labs(
    title = "Cross-Validation Performance Comparison",
    subtitle = "R-squared with Standard Error",
    x = "Model",
    y = "R-squared",
    fill = "Model"
  ) +
  theme_minimal()

print(p5)

# Step 6: Learning Curves
cat("Step 6: Creating learning curves\n")

# Function to create learning curve
create_learning_curve <- function(data, workflow, title) {
  
  sample_sizes <- seq(0.2, 1.0, by = 0.2)
  
  learning_data <- map_dfr(sample_sizes, function(prop) {
    
    # Sample training data
    sample_data <- data %>% slice_sample(prop = prop)
    
    # Fit model
    fit <- workflow %>% fit(sample_data)
    
    # Training performance
    train_pred <- fit %>% predict(sample_data) %>% bind_cols(sample_data)
    train_rsq <- train_pred %>% 
      rsq(truth = thermal_conductivity_W_per_mK, estimate = .pred) %>% 
      pull(.estimate)
    
    # Validation performance (on fixed test set)
    val_pred <- fit %>% predict(thermal_test) %>% bind_cols(thermal_test)
    val_rsq <- val_pred %>% 
      rsq(truth = thermal_conductivity_W_per_mK, estimate = .pred) %>% 
      pull(.estimate)
    
    tibble(
      sample_size = nrow(sample_data),
      training_rsq = train_rsq,
      validation_rsq = val_rsq
    )
  })
  
  learning_data %>%
    pivot_longer(cols = c(training_rsq, validation_rsq), 
                 names_to = "dataset", values_to = "rsq") %>%
    ggplot(aes(x = sample_size, y = rsq, color = dataset)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    labs(
      title = paste("Learning Curve:", title),
      x = "Training Set Size",
      y = "R-squared",
      color = "Dataset"
    ) +
    scale_color_manual(values = c("training_rsq" = "blue", "validation_rsq" = "red"),
                       labels = c("Training", "Validation")) +
    theme_minimal()
}

# Create learning curve for linear model
p6 <- create_learning_curve(thermal_train, thermal_workflow, "Linear Regression")
print(p6)

# Step 7: Feature Importance Visualization
cat("Step 7: Feature importance visualization\n")

# Extract and visualize feature importance
feature_importance <- thermal_fit %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  filter(term != "(Intercept)") %>%
  mutate(
    abs_estimate = abs(estimate),
    direction = ifelse(estimate > 0, "Positive", "Negative")
  ) %>%
  arrange(desc(abs_estimate))

p7 <- feature_importance %>%
  slice_head(n = 10) %>%
  ggplot(aes(x = reorder(term, abs_estimate), y = estimate, fill = direction)) +
  geom_col(alpha = 0.7) +
  coord_flip() +
  labs(
    title = "Feature Importance (Linear Model Coefficients)",
    x = "Features",
    y = "Coefficient Value",
    fill = "Effect"
  ) +
  scale_fill_manual(values = c("Positive" = "darkgreen", "Negative" = "darkred")) +
  theme_minimal()

print(p7)

# Step 8: Model Comparison Dashboard
cat("Step 8: Creating model comparison dashboard\n")

# Performance metrics comparison
performance_comparison <- cv_results %>%
  select(model, .metric, mean, std_err) %>%
  pivot_wider(names_from = .metric, values_from = c(mean, std_err))

cat("Model Performance Summary:\n")
print(performance_comparison)

# Create comprehensive comparison plot
p8 <- cv_results %>%
  ggplot(aes(x = model, y = mean, fill = model)) +
  geom_col(alpha = 0.7) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), width = 0.2) +
  facet_wrap(~.metric, scales = "free_y") +
  labs(
    title = "Comprehensive Model Performance Comparison",
    x = "Model",
    y = "Performance Metric",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

print(p8)

# Step 9: Prediction Interval Visualization
cat("Step 9: Prediction interval visualization\n")

# For demonstration, create prediction intervals using residual standard error
residual_se <- sqrt(mean(thermal_predictions$residuals^2))

prediction_intervals <- thermal_predictions %>%
  mutate(
    lower_bound = .pred - 1.96 * residual_se,
    upper_bound = .pred + 1.96 * residual_se,
    within_interval = thermal_conductivity_W_per_mK >= lower_bound & 
                     thermal_conductivity_W_per_mK <= upper_bound
  )

p9 <- prediction_intervals %>%
  ggplot(aes(x = .pred)) +
  geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound), alpha = 0.3, fill = "lightblue") +
  geom_point(aes(y = thermal_conductivity_W_per_mK, color = within_interval), alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Prediction Intervals (95% Confidence)",
    x = "Predicted Values",
    y = "Actual Values",
    color = "Within Interval"
  ) +
  scale_color_manual(values = c("TRUE" = "darkgreen", "FALSE" = "red")) +
  theme_minimal()

print(p9)

coverage_rate <- mean(prediction_intervals$within_interval)
cat("Prediction interval coverage rate: ", round(coverage_rate * 100, 1), "%\n\n")

# Step 10: Summary Visualization
cat("Step 10: Creating summary visualization\n")

# Create a summary metrics table
summary_metrics <- thermal_predictions %>%
  metrics(truth = thermal_conductivity_W_per_mK, estimate = .pred) %>%
  mutate(
    .estimate = round(.estimate, 4),
    interpretation = case_when(
      .metric == "rsq" ~ paste("Explains", round(.estimate * 100, 1), "% of variance"),
      .metric == "rmse" ~ paste("Average error:", round(.estimate, 2), "units"),
      .metric == "mae" ~ paste("Median error:", round(.estimate, 2), "units"),
      TRUE ~ ""
    )
  )

cat("Final Model Performance Summary:\n")
print(summary_metrics)

# Summary
cat("\n=== Visualization Summary ===\n")
cat("✓ Exploratory data analysis visualizations\n")
cat("✓ Correlation analysis across datasets\n")
cat("✓ Comprehensive model diagnostic plots\n")
cat("✓ Cross-validation results visualization\n")
cat("✓ Learning curve analysis\n")
cat("✓ Feature importance visualization\n")
cat("✓ Model comparison dashboard\n")
cat("✓ Prediction interval analysis\n")
cat("✓ Performance summary and interpretation\n\n")

cat("Key Visualization Concepts:\n")
cat("1. Always check residual patterns for model assumptions\n")
cat("2. Use Q-Q plots to assess normality of residuals\n")
cat("3. Learning curves help diagnose overfitting/underfitting\n")
cat("4. Feature importance guides model interpretation\n")
cat("5. Prediction intervals quantify uncertainty\n")
cat("6. Cross-validation provides robust performance estimates\n")
