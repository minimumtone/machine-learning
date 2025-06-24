# 01: Data Generation for Materials Engineering ML
# This script demonstrates how to generate and explore synthetic materials datasets

# Load required libraries
library(tidymodels)
library(dplyr)
library(ggplot2)
library(corrplot)

# Source utility functions
source("utils.R")

cat("=== Materials Engineering Data Generation ===\n\n")

# Generate three different materials datasets
cat("Generating synthetic materials datasets...\n")

# 1. Thermal Conductivity Dataset
thermal_data <- generate_thermal_conductivity_data(n_samples = 300, noise_level = 10)
cat("✓ Thermal conductivity dataset: ", nrow(thermal_data), " samples\n")

# 2. Electrical Conductivity Dataset  
electrical_data <- generate_electrical_conductivity_data(n_samples = 300, noise_level = 100)
cat("✓ Electrical conductivity dataset: ", nrow(electrical_data), " samples\n")

# 3. Mechanical Strength Dataset
strength_data <- generate_mechanical_strength_data(n_samples = 300, noise_level = 50)
cat("✓ Mechanical strength dataset: ", nrow(strength_data), " samples\n\n")

# Explore the thermal conductivity dataset in detail
cat("=== Thermal Conductivity Dataset Exploration ===\n")
cat("Dataset dimensions: ", dim(thermal_data)[1], " rows x ", dim(thermal_data)[2], " columns\n\n")

cat("Summary statistics:\n")
print(summary(thermal_data))
cat("\n")

cat("Correlation matrix:\n")
cor_matrix <- cor(thermal_data)
print(round(cor_matrix, 3))
cat("\n")

# Visualize relationships in thermal conductivity data
cat("Creating visualizations...\n")

# Pairwise scatter plots
p1 <- thermal_data %>%
  ggplot(aes(x = temperature_K, y = thermal_conductivity_W_per_mK)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(
    title = "Thermal Conductivity vs Temperature",
    x = "Temperature (K)",
    y = "Thermal Conductivity (W/m·K)"
  ) +
  theme_minimal()

p2 <- thermal_data %>%
  ggplot(aes(x = pressure_GPa, y = thermal_conductivity_W_per_mK)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(
    title = "Thermal Conductivity vs Pressure", 
    x = "Pressure (GPa)",
    y = "Thermal Conductivity (W/m·K)"
  ) +
  theme_minimal()

p3 <- thermal_data %>%
  ggplot(aes(x = processing_time_h, y = thermal_conductivity_W_per_mK)) +
  geom_point(alpha = 0.6, color = "purple") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(
    title = "Thermal Conductivity vs Processing Time",
    x = "Processing Time (hours)", 
    y = "Thermal Conductivity (W/m·K)"
  ) +
  theme_minimal()

# Display plots
print(p1)
print(p2) 
print(p3)

# Distribution plots
p4 <- thermal_data %>%
  ggplot(aes(x = thermal_conductivity_W_per_mK)) +
  geom_histogram(bins = 30, fill = "lightblue", alpha = 0.7, color = "black") +
  labs(
    title = "Distribution of Thermal Conductivity",
    x = "Thermal Conductivity (W/m·K)",
    y = "Frequency"
  ) +
  theme_minimal()

print(p4)

# Quick look at other datasets
cat("\n=== Quick Overview of Other Datasets ===\n")

cat("Electrical Conductivity Data:\n")
print(head(electrical_data, 3))
cat("Range of electrical conductivity: ", 
    round(min(electrical_data$electrical_conductivity_S_per_m), 1), " to ",
    round(max(electrical_data$electrical_conductivity_S_per_m), 1), " S/m\n\n")

cat("Mechanical Strength Data:\n") 
print(head(strength_data, 3))
cat("Range of mechanical strength: ",
    round(min(strength_data$mechanical_strength_MPa), 1), " to ",
    round(max(strength_data$mechanical_strength_MPa), 1), " MPa\n\n")

# Save datasets for use in other scripts
cat("Saving datasets to RDS files...\n")
saveRDS(thermal_data, "thermal_conductivity_data.rds")
saveRDS(electrical_data, "electrical_conductivity_data.rds") 
saveRDS(strength_data, "mechanical_strength_data.rds")

cat("✓ Data generation complete!\n")
cat("✓ Datasets saved as RDS files\n")
cat("✓ Ready for machine learning workflows\n\n")

cat("Key Takeaways:\n")
cat("1. Synthetic data mimics realistic materials engineering relationships\n")
cat("2. Features show expected correlations with target properties\n") 
cat("3. Noise levels reflect typical experimental uncertainty\n")
cat("4. Data is ready for train/test splitting and model development\n")
