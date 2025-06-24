# Utility Functions for Materials Engineering ML Examples
# This file contains helper functions for data generation and model evaluation

library(tidymodels)
library(dplyr)
library(ggplot2)

#' Generate synthetic thermal conductivity data
#' 
#' Creates realistic thermal conductivity dataset based on processing conditions
#' 
#' @param n_samples Number of samples to generate
#' @param noise_level Standard deviation of noise to add
#' @param random_seed Random seed for reproducibility
#' @return Data frame with processing conditions and thermal conductivity
generate_thermal_conductivity_data <- function(n_samples = 200, noise_level = 10, random_seed = 42) {
  set.seed(random_seed)
  
  # Processing conditions (realistic ranges for ceramics/metals)
  temperature <- runif(n_samples, 800, 1400)  # Kelvin
  pressure <- runif(n_samples, 1.0, 5.0)     # GPa
  time <- runif(n_samples, 1.0, 10.0)        # hours
  
  # Realistic relationship: higher temp and pressure increase conductivity
  thermal_conductivity <- 50 + 0.1 * temperature + 20 * pressure + 5 * time + 
                         rnorm(n_samples, 0, noise_level)
  
  data.frame(
    temperature_K = temperature,
    pressure_GPa = pressure,
    processing_time_h = time,
    thermal_conductivity_W_per_mK = thermal_conductivity
  )
}

#' Generate synthetic electrical conductivity data
#' 
#' Creates realistic electrical conductivity dataset for alloy systems
#' 
#' @param n_samples Number of samples to generate  
#' @param noise_level Standard deviation of noise to add
#' @param random_seed Random seed for reproducibility
#' @return Data frame with composition and electrical conductivity
generate_electrical_conductivity_data <- function(n_samples = 200, noise_level = 100, random_seed = 42) {
  set.seed(random_seed)
  
  # Composition and processing variables
  composition <- runif(n_samples, 0.1, 0.9)      # Alloy fraction
  annealing_temp <- runif(n_samples, 600, 1200)  # Kelvin
  grain_size <- runif(n_samples, 1, 50)          # micrometers
  
  # Realistic relationship: composition dominates, grain size and temp matter
  electrical_conductivity <- 1000 * composition + 0.5 * annealing_temp + 
                           10 * grain_size + rnorm(n_samples, 0, noise_level)
  
  data.frame(
    alloy_composition = composition,
    annealing_temperature_K = annealing_temp,
    grain_size_um = grain_size,
    electrical_conductivity_S_per_m = electrical_conductivity
  )
}

#' Generate synthetic mechanical strength data
#' 
#' Creates realistic mechanical strength dataset with complex relationships
#' 
#' @param n_samples Number of samples to generate
#' @param noise_level Standard deviation of noise to add  
#' @param random_seed Random seed for reproducibility
#' @return Data frame with microstructure and mechanical strength
generate_mechanical_strength_data <- function(n_samples = 200, noise_level = 50, random_seed = 42) {
  set.seed(random_seed)
  
  # Microstructural variables
  grain_size <- runif(n_samples, 1, 100)         # micrometers
  heat_treatment_temp <- runif(n_samples, 400, 800)  # Celsius
  cooling_rate <- runif(n_samples, 1, 100)       # K/min
  
  # Hall-Petch relationship: strength inversely related to sqrt(grain_size)
  # Plus effects of heat treatment
  mechanical_strength <- 200 + 500 / sqrt(grain_size) + 
                        0.3 * heat_treatment_temp + 2 * cooling_rate +
                        rnorm(n_samples, 0, noise_level)
  
  data.frame(
    grain_size_um = grain_size,
    heat_treatment_temp_C = heat_treatment_temp,
    cooling_rate_K_per_min = cooling_rate,
    mechanical_strength_MPa = mechanical_strength
  )
}

#' Calculate overfitting metrics
#' 
#' Compares training and validation performance to detect overfitting
#' 
#' @param train_metrics Training set performance metrics
#' @param val_metrics Validation set performance metrics
#' @return Data frame with overfitting indicators
calculate_overfitting_metrics <- function(train_metrics, val_metrics) {
  # Extract R-squared values
  train_rsq <- train_metrics %>% 
    filter(.metric == "rsq") %>% 
    pull(.estimate)
  
  val_rsq <- val_metrics %>% 
    filter(.metric == "rsq") %>% 
    pull(.estimate)
  
  # Calculate overfitting indicator
  overfitting_indicator <- train_rsq - val_rsq
  
  data.frame(
    train_rsq = train_rsq,
    validation_rsq = val_rsq,
    overfitting_indicator = overfitting_indicator,
    overfitting_severity = case_when(
      overfitting_indicator < 0.05 ~ "None",
      overfitting_indicator < 0.15 ~ "Mild", 
      overfitting_indicator < 0.25 ~ "Moderate",
      TRUE ~ "Severe"
    )
  )
}

#' Create model comparison plot
#' 
#' Visualizes performance of multiple models with error bars
#' 
#' @param results_df Data frame with model results from cross-validation
#' @param metric_name Name of metric to plot (e.g., "rsq", "rmse")
#' @return ggplot object
plot_model_comparison <- function(results_df, metric_name = "rsq") {
  results_df %>%
    filter(.metric == metric_name) %>%
    ggplot(aes(x = model_name, y = mean, fill = model_name)) +
    geom_col(alpha = 0.7) +
    geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), 
                  width = 0.2) +
    labs(
      title = paste("Model Comparison:", toupper(metric_name)),
      x = "Model Type",
      y = paste(toupper(metric_name), "Score"),
      fill = "Model"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

#' Detect prediction problems
#' 
#' Identifies common prediction issues like single-value or step-like predictions
#' 
#' @param predictions Vector of model predictions
#' @return Character vector describing any problems found
detect_prediction_problems <- function(predictions) {
  problems <- character(0)
  
  # Check for single-value predictions (all predictions very similar)
  if (sd(predictions, na.rm = TRUE) < 0.01 * mean(abs(predictions), na.rm = TRUE)) {
    problems <- c(problems, "Single-value predictions")
  }
  
  # Check for step-like predictions (only a few distinct values)
  unique_vals <- length(unique(round(predictions, 2)))
  if (unique_vals < 0.1 * length(predictions) && unique_vals < 10) {
    problems <- c(problems, "Step-like predictions")
  }
  
  # Check for extreme predictions
  q99 <- quantile(predictions, 0.99, na.rm = TRUE)
  q01 <- quantile(predictions, 0.01, na.rm = TRUE)
  if (any(predictions > 3 * q99 | predictions < 3 * q01, na.rm = TRUE)) {
    problems <- c(problems, "Extreme predictions")
  }
  
  if (length(problems) == 0) {
    return("None")
  } else {
    return(paste(problems, collapse = ", "))
  }
}
