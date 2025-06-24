# R tidymodels Examples for Materials Engineering Machine Learning

This collection provides comprehensive examples of machine learning applications in materials engineering using the R tidymodels ecosystem. The examples demonstrate proper data splitting, cross-validation, model comparison, and generalization performance evaluation.

## Overview

Materials engineering involves predicting material properties from processing conditions and composition. This collection covers:

- **Thermal Conductivity Prediction**: Predicting thermal conductivity from temperature, pressure, and processing time
- **Electrical Conductivity Prediction**: Predicting electrical conductivity from composition and annealing conditions  
- **Mechanical Strength Prediction**: Predicting mechanical strength from grain size and heat treatment parameters

## Key Learning Objectives

1. **Proper Data Splitting**: Using `rsample` for training/validation/test splits
2. **Cross-Validation**: Implementing k-fold CV with `tune` and `workflows`
3. **Model Comparison**: Comparing linear regression, ridge regression, polynomial features, and random forest
4. **Overfitting Detection**: Identifying when models perform well on training but poorly on validation data
5. **Generalization Assessment**: Evaluating true model performance on unseen data
6. **Performance Metrics**: Understanding R², RMSE, MAE in materials context

## Files Structure

- `01_data_generation.R`: Generate synthetic materials engineering datasets
- `02_basic_workflow.R`: Basic tidymodels workflow with train/test split
- `03_cross_validation.R`: Cross-validation and model tuning examples
- `04_model_comparison.R`: Compare multiple model types and detect overfitting
- `05_advanced_features.R`: Polynomial features and feature engineering
- `06_visualization.R`: Plotting results and model diagnostics
- `utils.R`: Helper functions for data generation and evaluation

## Required Packages

```r
# Core tidymodels
library(tidymodels)
library(recipes)
library(parsnip)
library(workflows)
library(tune)
library(yardstick)
library(rsample)

# Data manipulation and visualization
library(dplyr)
library(ggplot2)
library(purrr)

# Additional models
library(randomForest)
library(glmnet)
```

## Getting Started

1. Install required packages
2. Run scripts in order (01 through 06)
3. Each script builds on previous concepts
4. Pay attention to comments explaining key concepts

## Key Concepts Demonstrated

### Training/Validation Data Separation
- Proper use of `initial_split()` and `training()`/`testing()`
- Importance of never touching test data until final evaluation
- Using validation sets for model selection

### Cross-Validation for Model Evaluation
- k-fold cross-validation with `vfold_cv()`
- Nested resampling for hyperparameter tuning
- Understanding bias-variance tradeoff

### Overfitting Detection
- Comparing training vs validation performance
- Identifying models that memorize rather than generalize
- Using learning curves to diagnose overfitting

### Model Comparison Framework
- Fair comparison using same data splits
- Statistical significance testing
- Ranking models by generalization performance

## Materials Engineering Context

These examples use realistic materials engineering scenarios:

- **Processing Parameters**: Temperature, pressure, time, atmosphere
- **Composition Variables**: Alloy percentages, dopant concentrations
- **Microstructure Features**: Grain size, phase fractions, defect density
- **Target Properties**: Thermal/electrical conductivity, mechanical strength, corrosion resistance

The synthetic datasets include realistic noise levels and relationships commonly found in materials research.
