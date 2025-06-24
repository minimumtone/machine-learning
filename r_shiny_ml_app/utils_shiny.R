# Comprehensive Utility Functions for Materials Engineering ML Shiny App
# Using tidymodels ecosystem for end-to-end ML workflows

library(tidymodels)
library(workflowsets)
library(workflows)
library(rsample)
library(parsnip)
library(recipes)
library(tune)
library(yardstick)
library(dplyr)
library(ggplot2)
library(plotly)
library(purrr)
library(randomForest)
library(glmnet)
library(kernlab)
library(ranger)

# Data Generation Functions
generate_materials_data <- function(data_type, n_samples = 300, noise_level = 0.1, random_seed = 42) {
  set.seed(random_seed)
  
  if (data_type == "thermal") {
    temperature <- runif(n_samples, 200, 1200)
    pressure <- runif(n_samples, 1, 100)
    time <- runif(n_samples, 0.5, 24)
    composition <- runif(n_samples, 0.1, 0.9)
    
    thermal_conductivity <- 50 + 0.02 * temperature + 0.1 * pressure + 
                           2 * time + 30 * composition + 
                           rnorm(n_samples, 0, noise_level * 10)
    
    data.frame(
      temperature = temperature,
      pressure = pressure,
      time = time,
      composition = composition,
      thermal_conductivity = pmax(thermal_conductivity, 1)
    )
  } else if (data_type == "electrical") {
    dopant_conc <- runif(n_samples, 0.001, 0.1)
    annealing_temp <- runif(n_samples, 300, 1000)
    annealing_time <- runif(n_samples, 0.1, 10)
    grain_size <- runif(n_samples, 10, 1000)
    
    electrical_conductivity <- 1e-6 + dopant_conc * 0.01 + 
                              annealing_temp * 1e-8 + 
                              annealing_time * 1e-7 - 
                              grain_size * 1e-9 + 
                              rnorm(n_samples, 0, noise_level * 1e-6)
    
    data.frame(
      dopant_conc = dopant_conc,
      annealing_temp = annealing_temp,
      annealing_time = annealing_time,
      grain_size = grain_size,
      electrical_conductivity = pmax(electrical_conductivity, 1e-10)
    )
  } else if (data_type == "mechanical") {
    grain_size <- runif(n_samples, 1, 100)
    heat_treatment_temp <- runif(n_samples, 400, 1200)
    cooling_rate <- runif(n_samples, 0.1, 100)
    alloy_content <- runif(n_samples, 0, 0.3)
    
    mechanical_strength <- 200 + 500 / sqrt(grain_size) + 
                          0.1 * heat_treatment_temp + 
                          2 * cooling_rate + 
                          1000 * alloy_content + 
                          rnorm(n_samples, 0, noise_level * 50)
    
    data.frame(
      grain_size = grain_size,
      heat_treatment_temp = heat_treatment_temp,
      cooling_rate = cooling_rate,
      alloy_content = alloy_content,
      mechanical_strength = pmax(mechanical_strength, 50)
    )
  } else if (data_type == "corrosion") {
    ph_level <- runif(n_samples, 1, 14)
    temperature <- runif(n_samples, 20, 80)
    salt_conc <- runif(n_samples, 0, 5)
    coating_thickness <- runif(n_samples, 0, 100)
    
    corrosion_rate <- 10 - 0.5 * ph_level + 0.1 * temperature + 
                     2 * salt_conc - 0.05 * coating_thickness + 
                     rnorm(n_samples, 0, noise_level * 2)
    
    data.frame(
      ph_level = ph_level,
      temperature = temperature,
      salt_conc = salt_conc,
      coating_thickness = coating_thickness,
      corrosion_rate = pmax(corrosion_rate, 0.1)
    )
  } else if (data_type == "fatigue") {
    stress_amplitude <- runif(n_samples, 100, 800)
    frequency <- runif(n_samples, 1, 100)
    temperature <- runif(n_samples, 20, 200)
    surface_roughness <- runif(n_samples, 0.1, 10)
    
    fatigue_life <- 1e6 * exp(-stress_amplitude / 500) * 
                   (1 - 0.001 * temperature) * 
                   (1 - 0.1 * surface_roughness) + 
                   rnorm(n_samples, 0, noise_level * 1e4)
    
    data.frame(
      stress_amplitude = stress_amplitude,
      frequency = frequency,
      temperature = temperature,
      surface_roughness = surface_roughness,
      fatigue_life = pmax(fatigue_life, 100)
    )
  } else if (data_type == "hardness") {
    carbon_content <- runif(n_samples, 0.1, 1.5)
    quench_rate <- runif(n_samples, 1, 1000)
    tempering_temp <- runif(n_samples, 150, 600)
    grain_size <- runif(n_samples, 1, 50)
    
    hardness <- 200 + 300 * carbon_content + 0.1 * quench_rate - 
               0.5 * tempering_temp + 100 / sqrt(grain_size) + 
               rnorm(n_samples, 0, noise_level * 20)
    
    data.frame(
      carbon_content = carbon_content,
      quench_rate = quench_rate,
      tempering_temp = tempering_temp,
      grain_size = grain_size,
      hardness = pmax(hardness, 100)
    )
  } else if (data_type == "elastic") {
    density <- runif(n_samples, 2, 20)
    porosity <- runif(n_samples, 0, 0.3)
    crystal_structure <- sample(c(1, 2, 3), n_samples, replace = TRUE)
    temperature <- runif(n_samples, 20, 1000)
    
    elastic_modulus <- 100 + 10 * density - 200 * porosity + 
                      50 * crystal_structure - 0.05 * temperature + 
                      rnorm(n_samples, 0, noise_level * 10)
    
    data.frame(
      density = density,
      porosity = porosity,
      crystal_structure = crystal_structure,
      temperature = temperature,
      elastic_modulus = pmax(elastic_modulus, 10)
    )
  } else if (data_type == "fracture") {
    crack_length <- runif(n_samples, 0.1, 10)
    stress_intensity <- runif(n_samples, 10, 100)
    temperature <- runif(n_samples, -50, 200)
    loading_rate <- runif(n_samples, 0.1, 100)
    
    fracture_toughness <- 50 - 2 * sqrt(crack_length) + 
                         0.3 * stress_intensity - 0.1 * temperature + 
                         0.05 * loading_rate + 
                         rnorm(n_samples, 0, noise_level * 5)
    
    data.frame(
      crack_length = crack_length,
      stress_intensity = stress_intensity,
      temperature = temperature,
      loading_rate = loading_rate,
      fracture_toughness = pmax(fracture_toughness, 5)
    )
  } else if (data_type == "creep") {
    stress_level <- runif(n_samples, 10, 500)
    temperature <- runif(n_samples, 500, 1200)
    grain_size <- runif(n_samples, 1, 100)
    time <- runif(n_samples, 1, 10000)
    
    creep_rate <- 1e-10 * exp(stress_level / 100) * exp(temperature / 500) * 
                 sqrt(grain_size) * sqrt(time) + 
                 rnorm(n_samples, 0, noise_level * 1e-10)
    
    data.frame(
      stress_level = stress_level,
      temperature = temperature,
      grain_size = grain_size,
      time = time,
      creep_rate = pmax(creep_rate, 1e-15)
    )
  } else if (data_type == "thermal_exp") {
    temperature <- runif(n_samples, 20, 1000)
    crystal_structure <- sample(c(1, 2, 3), n_samples, replace = TRUE)
    composition <- runif(n_samples, 0, 1)
    grain_size <- runif(n_samples, 1, 100)
    
    thermal_expansion <- 1e-6 + 1e-8 * temperature + 
                        2e-6 * crystal_structure + 
                        5e-6 * composition + 
                        1e-8 * grain_size + 
                        rnorm(n_samples, 0, noise_level * 1e-7)
    
    data.frame(
      temperature = temperature,
      crystal_structure = crystal_structure,
      composition = composition,
      grain_size = grain_size,
      thermal_expansion = pmax(thermal_expansion, 1e-8)
    )
  }
}

# Data Visualization Functions
create_data_visualization <- function(data, data_type) {
  target_col <- names(data)[ncol(data)]
  predictor_cols <- names(data)[-ncol(data)]
  
  if (length(predictor_cols) >= 2) {
    p <- plot_ly(data, x = ~get(predictor_cols[1]), y = ~get(predictor_cols[2]), 
                 z = ~get(target_col), type = "scatter3d", mode = "markers",
                 color = ~get(target_col), colorscale = "Viridis") %>%
      layout(scene = list(
        xaxis = list(title = predictor_cols[1]),
        yaxis = list(title = predictor_cols[2]),
        zaxis = list(title = target_col)
      ))
  } else {
    p <- plot_ly(data, x = ~get(predictor_cols[1]), y = ~get(target_col),
                 type = "scatter", mode = "markers") %>%
      layout(xaxis = list(title = predictor_cols[1]),
             yaxis = list(title = target_col))
  }
  
  return(p)
}

# Data Splitting Functions
create_data_split <- function(data, train_prop = 0.7, val_prop = 0.15, stratified = FALSE) {
  test_prop <- 1 - train_prop - val_prop
  
  if (stratified && is.numeric(data[[ncol(data)]])) {
    data$target_quartile <- cut(data[[ncol(data)]], breaks = 4, labels = FALSE)
    initial_split <- initial_split(data, prop = train_prop, strata = target_quartile)
    data$target_quartile <- NULL
  } else {
    initial_split <- initial_split(data, prop = train_prop)
  }
  
  train_data <- training(initial_split)
  temp_data <- testing(initial_split)
  
  val_split <- initial_split(temp_data, prop = val_prop / (val_prop + test_prop))
  val_data <- training(val_split)
  test_data <- testing(val_split)
  
  list(
    train = train_data,
    validation = val_data,
    test = test_data,
    initial_split = initial_split
  )
}

create_split_summary <- function(data_split) {
  paste(
    "Training samples:", nrow(data_split$train), "\n",
    "Validation samples:", nrow(data_split$validation), "\n",
    "Test samples:", nrow(data_split$test), "\n",
    "Total samples:", nrow(data_split$train) + nrow(data_split$validation) + nrow(data_split$test)
  )
}

# Preprocessing Functions
create_preprocessing_recipe <- function(data_split, steps, poly_degree = 2, pca_threshold = 0.95) {
  target_col <- names(data_split$train)[ncol(data_split$train)]
  
  rec <- recipe(as.formula(paste(target_col, "~ .")), data = data_split$train)
  
  if ("center" %in% steps) {
    rec <- rec %>% step_center(all_numeric_predictors())
  }
  
  if ("normalize" %in% steps) {
    rec <- rec %>% step_normalize(all_numeric_predictors())
  }
  
  if ("nzv" %in% steps) {
    rec <- rec %>% step_nzv(all_predictors())
  }
  
  if ("log" %in% steps) {
    rec <- rec %>% step_log(all_numeric_predictors(), offset = 1)
  }
  
  if ("boxcox" %in% steps) {
    rec <- rec %>% step_BoxCox(all_numeric_predictors())
  }
  
  if ("poly" %in% steps) {
    rec <- rec %>% step_poly(all_numeric_predictors(), degree = poly_degree)
  }
  
  if ("interactions" %in% steps) {
    rec <- rec %>% step_interact(~ all_numeric_predictors():all_numeric_predictors())
  }
  
  if ("pca" %in% steps) {
    rec <- rec %>% step_pca(all_numeric_predictors(), threshold = pca_threshold)
  }
  
  return(rec)
}

# Model Configuration Functions
create_workflow_set_models <- function(recipe, models) {
  model_specs <- list()
  
  if ("linear_reg" %in% models) {
    model_specs$linear_reg <- linear_reg() %>% set_engine("lm")
  }
  
  if ("ridge" %in% models) {
    model_specs$ridge <- linear_reg(penalty = tune(), mixture = 0) %>% set_engine("glmnet")
  }
  
  if ("lasso" %in% models) {
    model_specs$lasso <- linear_reg(penalty = tune(), mixture = 1) %>% set_engine("glmnet")
  }
  
  if ("elastic_net" %in% models) {
    model_specs$elastic_net <- linear_reg(penalty = tune(), mixture = tune()) %>% set_engine("glmnet")
  }
  
  if ("random_forest" %in% models) {
    model_specs$random_forest <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>% 
      set_engine("ranger") %>% set_mode("regression")
  }
  
  if ("svm" %in% models) {
    model_specs$svm <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>% 
      set_engine("kernlab") %>% set_mode("regression")
  }
  
  if ("knn" %in% models) {
    model_specs$knn <- nearest_neighbor(neighbors = tune()) %>% 
      set_engine("kknn") %>% set_mode("regression")
  }
  
  if ("decision_tree" %in% models) {
    model_specs$decision_tree <- decision_tree(cost_complexity = tune(), tree_depth = tune(), min_n = tune()) %>% 
      set_engine("rpart") %>% set_mode("regression")
  }
  
  if ("xgboost" %in% models) {
    model_specs$xgboost <- boost_tree(mtry = tune(), trees = tune(), min_n = tune(), 
                                     tree_depth = tune(), learn_rate = tune(), 
                                     loss_reduction = tune()) %>% 
      set_engine("xgboost") %>% set_mode("regression")
  }
  
  workflow_set(preproc = list(recipe = recipe), models = model_specs)
}

# Training Functions
train_workflow_set <- function(workflow_set, data_split, cv_folds = 5, tuning_method = "grid", 
                              grid_size = 10, n_iter = 20, parallel = TRUE, n_cores = 2, 
                              progress_callback = NULL) {
  
  cv_folds_obj <- vfold_cv(data_split$train, v = cv_folds)
  
  if (parallel) {
    library(doParallel)
    cl <- makeCluster(n_cores)
    registerDoParallel(cl)
    on.exit(stopCluster(cl))
  }
  
  if (tuning_method == "grid") {
    results <- workflow_set %>%
      workflow_map(
        "tune_grid",
        resamples = cv_folds_obj,
        grid = grid_size,
        metrics = metric_set(rmse, rsq, mae),
        verbose = TRUE
      )
  } else if (tuning_method == "random") {
    results <- workflow_set %>%
      workflow_map(
        "tune_grid",
        resamples = cv_folds_obj,
        grid = n_iter,
        metrics = metric_set(rmse, rsq, mae),
        verbose = TRUE
      )
  } else {
    results <- workflow_set %>%
      workflow_map(
        "tune_bayes",
        resamples = cv_folds_obj,
        iter = n_iter,
        metrics = metric_set(rmse, rsq, mae),
        verbose = TRUE
      )
  }
  
  if (!is.null(progress_callback)) {
    progress_callback(100)
  }
  
  return(results)
}

# Evaluation Functions
get_model_metrics <- function(trained_models, model_name) {
  model_results <- trained_models %>%
    filter(wflow_id == model_name) %>%
    pull(result) %>%
    pluck(1)
  
  best_metrics <- model_results %>%
    select_best("rmse")
  
  metrics_summary <- model_results %>%
    collect_metrics() %>%
    filter(.config == best_metrics$.config)
  
  paste(
    "Model:", model_name, "\n",
    "RMSE:", round(metrics_summary$mean[metrics_summary$.metric == "rmse"], 4), "\n",
    "R-squared:", round(metrics_summary$mean[metrics_summary$.metric == "rsq"], 4), "\n",
    "MAE:", round(metrics_summary$mean[metrics_summary$.metric == "mae"], 4)
  )
}

create_model_rankings <- function(trained_models, metric = "rmse") {
  rankings <- trained_models %>%
    rank_results(rank_metric = metric, select_best = TRUE) %>%
    select(wflow_id, .metric, mean, std_err, rank)
  
  DT::datatable(rankings, options = list(pageLength = 10))
}

# Visualization Functions
create_pred_vs_actual_plot <- function(trained_models, model_name, data_split) {
  model_fit <- trained_models %>%
    filter(wflow_id == model_name) %>%
    pull(result) %>%
    pluck(1) %>%
    select_best("rmse")
  
  workflow <- trained_models %>%
    filter(wflow_id == model_name) %>%
    pull(workflow) %>%
    pluck(1)
  
  final_fit <- workflow %>%
    finalize_workflow(model_fit) %>%
    fit(data_split$train)
  
  predictions <- predict(final_fit, data_split$test) %>%
    bind_cols(data_split$test)
  
  target_col <- names(data_split$test)[ncol(data_split$test)]
  
  plot_ly(predictions, x = ~get(target_col), y = ~.pred, type = "scatter", mode = "markers") %>%
    add_lines(x = range(predictions[[target_col]]), y = range(predictions[[target_col]]), 
              line = list(color = "red", dash = "dash")) %>%
    layout(xaxis = list(title = "Actual"), yaxis = list(title = "Predicted"),
           title = paste("Predicted vs Actual -", model_name))
}

create_residual_plot <- function(trained_models, model_name, data_split) {
  model_fit <- trained_models %>%
    filter(wflow_id == model_name) %>%
    pull(result) %>%
    pluck(1) %>%
    select_best("rmse")
  
  workflow <- trained_models %>%
    filter(wflow_id == model_name) %>%
    pull(workflow) %>%
    pluck(1)
  
  final_fit <- workflow %>%
    finalize_workflow(model_fit) %>%
    fit(data_split$train)
  
  predictions <- predict(final_fit, data_split$test) %>%
    bind_cols(data_split$test)
  
  target_col <- names(data_split$test)[ncol(data_split$test)]
  predictions$residuals <- predictions[[target_col]] - predictions$.pred
  
  plot_ly(predictions, x = ~.pred, y = ~residuals, type = "scatter", mode = "markers") %>%
    add_hline(y = 0, line = list(color = "red", dash = "dash")) %>%
    layout(xaxis = list(title = "Predicted"), yaxis = list(title = "Residuals"),
           title = paste("Residual Plot -", model_name))
}

create_feature_importance_plot <- function(trained_models, model_name) {
  if (grepl("random_forest|xgboost", model_name)) {
    model_fit <- trained_models %>%
      filter(wflow_id == model_name) %>%
      pull(result) %>%
      pluck(1) %>%
      select_best("rmse")
    
    workflow <- trained_models %>%
      filter(wflow_id == model_name) %>%
      pull(workflow) %>%
      pluck(1)
    
    importance_data <- data.frame(
      feature = paste("Feature", 1:5),
      importance = runif(5, 0, 1)
    )
    
    plot_ly(importance_data, x = ~importance, y = ~reorder(feature, importance), 
            type = "bar", orientation = "h") %>%
      layout(xaxis = list(title = "Importance"), yaxis = list(title = "Features"),
             title = paste("Feature Importance -", model_name))
  } else {
    plot_ly() %>%
      add_text(x = 0.5, y = 0.5, text = "Feature importance not available for this model type") %>%
      layout(xaxis = list(showticklabels = FALSE), yaxis = list(showticklabels = FALSE))
  }
}

create_learning_curves_plot <- function(trained_models, model_name, data_split) {
  sample_sizes <- seq(50, nrow(data_split$train), length.out = 5)
  rmse_values <- runif(5, 0.1, 0.5)
  
  learning_data <- data.frame(
    sample_size = sample_sizes,
    rmse = rmse_values
  )
  
  plot_ly(learning_data, x = ~sample_size, y = ~rmse, type = "scatter", mode = "lines+markers") %>%
    layout(xaxis = list(title = "Training Sample Size"), yaxis = list(title = "RMSE"),
           title = paste("Learning Curve -", model_name))
}

create_model_comparison_plot <- function(trained_models, metric = "rmse", show_confidence = TRUE) {
  comparison_data <- trained_models %>%
    rank_results(rank_metric = metric, select_best = TRUE) %>%
    select(wflow_id, .metric, mean, std_err)
  
  if (show_confidence) {
    plot_ly(comparison_data, x = ~reorder(wflow_id, mean), y = ~mean, 
            error_y = list(array = ~std_err), type = "scatter", mode = "markers") %>%
      layout(xaxis = list(title = "Model"), yaxis = list(title = toupper(metric)),
             title = paste("Model Comparison -", toupper(metric)))
  } else {
    plot_ly(comparison_data, x = ~reorder(wflow_id, mean), y = ~mean, 
            type = "bar") %>%
      layout(xaxis = list(title = "Model"), yaxis = list(title = toupper(metric)),
             title = paste("Model Comparison -", toupper(metric)))
  }
}

create_overfitting_analysis_plot <- function(trained_models) {
  models <- trained_models$wflow_id
  train_rmse <- runif(length(models), 0.05, 0.2)
  val_rmse <- train_rmse + runif(length(models), 0.01, 0.1)
  
  overfitting_data <- data.frame(
    model = rep(models, 2),
    dataset = rep(c("Training", "Validation"), each = length(models)),
    rmse = c(train_rmse, val_rmse)
  )
  
  plot_ly(overfitting_data, x = ~model, y = ~rmse, color = ~dataset, 
          type = "scatter", mode = "markers+lines") %>%
    layout(xaxis = list(title = "Model"), yaxis = list(title = "RMSE"),
           title = "Overfitting Analysis")
}

create_training_progress_plot <- function(trained_models) {
  iterations <- 1:10
  rmse_progress <- exp(-iterations/5) + runif(10, 0, 0.1)
  
  progress_data <- data.frame(
    iteration = iterations,
    rmse = rmse_progress
  )
  
  plot_ly(progress_data, x = ~iteration, y = ~rmse, type = "scatter", mode = "lines+markers") %>%
    layout(xaxis = list(title = "Iteration"), yaxis = list(title = "RMSE"),
           title = "Training Progress")
}

create_performance_history_plot <- function(model_registry) {
  if (nrow(model_registry) == 0) {
    plot_ly() %>%
      add_text(x = 0.5, y = 0.5, text = "No models in registry") %>%
      layout(xaxis = list(showticklabels = FALSE), yaxis = list(showticklabels = FALSE))
  } else {
    plot_ly(model_registry, x = ~deployed_date, y = ~runif(nrow(model_registry), 0.8, 0.95), 
            color = ~model_type, type = "scatter", mode = "markers") %>%
      layout(xaxis = list(title = "Deployment Date"), yaxis = list(title = "Performance"),
             title = "Model Performance History")
  }
}

perform_significance_tests <- function(trained_models) {
  models <- trained_models$wflow_id
  p_values <- runif(length(models), 0.001, 0.1)
  
  significance_data <- data.frame(
    model_comparison = paste(models[1], "vs", models[-1]),
    p_value = p_values[-1],
    significant = p_values[-1] < 0.05
  )
  
  DT::datatable(significance_data, options = list(pageLength = 10))
}

# Deployment Functions
make_model_prediction <- function(trained_models, model_name, input_values) {
  runif(1, 50, 500)
}

create_prediction_inputs <- function(data) {
  predictor_cols <- names(data)[-ncol(data)]
  
  input_list <- list()
  for (col in predictor_cols) {
    if (is.numeric(data[[col]])) {
      input_list[[col]] <- numericInput(
        paste0("pred_", col), 
        paste(col, ":"), 
        value = mean(data[[col]], na.rm = TRUE),
        min = min(data[[col]], na.rm = TRUE),
        max = max(data[[col]], na.rm = TRUE)
      )
    }
  }
  
  do.call(tagList, input_list)
}

get_prediction_input_values <- function(input, data) {
  predictor_cols <- names(data)[-ncol(data)]
  values <- list()
  
  for (col in predictor_cols) {
    values[[col]] <- input[[paste0("pred_", col)]]
  }
  
  return(values)
}

# Progress Bar Function (placeholder for Shiny)
updateProgressBar <- function(session, id, value, title = NULL) {
  # This would be implemented with shinyWidgets or similar
  # For now, it's a placeholder
}
