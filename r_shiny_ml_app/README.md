# Materials Engineering Machine Learning Shiny Application

A comprehensive R Shiny application for materials engineering machine learning using the tidymodels ecosystem. This application provides an end-to-end workflow from data generation through model deployment with 30+ diverse materials engineering scenarios.

## Features

### 🔬 Materials Engineering Scenarios (30+)
- **Alloy Development**: Steel, Aluminum, Titanium, Superalloys
- **Ceramic Processing**: Alumina, Silicon Carbide, Zirconia, Piezoelectric
- **Polymer Engineering**: Thermoplastic, Thermoset, Blends, Biodegradable
- **Composite Materials**: Carbon Fiber, Glass Fiber, Natural Fiber, Metal Matrix
- **Nanomaterials**: Nanoparticles, Carbon Nanotubes, Graphene, Quantum Dots
- **Biomaterials**: Implants, Drug Delivery, Tissue Scaffolds, Dental Materials
- **Electronic Materials**: Semiconductors, Conductive Polymers, Dielectrics, Magnetics
- **Energy Materials**: Battery Electrodes, Solar Cells, Fuel Cells, Thermoelectrics

### 🧠 Machine Learning Models
- Linear Regression, Ridge, Lasso, Elastic Net
- Random Forest, SVM, KNN, Neural Network
- XGBoost, Decision Tree

### 📊 Comprehensive Workflow
1. **Data Generation**: Synthetic materials datasets
2. **Scenario Selection**: 30+ materials engineering scenarios
3. **Data Preprocessing**: Feature engineering, scaling, transformations
4. **Model Configuration**: Workflow sets with multiple models
5. **Training & Tuning**: Hyperparameter optimization
6. **Model Evaluation**: Performance metrics and diagnostics
7. **Model Comparison**: Statistical comparison across models
8. **Deployment**: Model prediction interface
9. **Model Management**: Registry and performance tracking

### 🛠 tidymodels Integration
- **workflows**: Model workflow management
- **workflowsets**: Multiple model comparison
- **rsample**: Data splitting and resampling
- **parsnip**: Model specification
- **recipes**: Feature engineering
- **tune**: Hyperparameter tuning
- **yardstick**: Performance metrics

## Installation

```r
install.packages(c(
  "shiny", "shinydashboard", "DT", "plotly",
  "tidymodels", "workflowsets", "workflows", 
  "rsample", "parsnip", "recipes", "tune", "yardstick",
  "dplyr", "ggplot2", "purrr",
  "randomForest", "glmnet", "kernlab", "ranger"
))
```

## Usage

```r
shiny::runApp("r_shiny_ml_app")
```

## Application Structure

- `app.R`: Main Shiny application with UI and server logic
- `utils_shiny.R`: Utility functions for data processing and modeling
- `scenarios.R`: Materials engineering scenarios and data generation
- `README.md`: Documentation and usage instructions
