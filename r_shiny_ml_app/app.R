# Comprehensive Materials Engineering Machine Learning Shiny Application
# Using tidymodels ecosystem for end-to-end ML workflows

library(shiny)
library(shinydashboard)
library(DT)
library(plotly)
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
library(purrr)
library(randomForest)
library(glmnet)
library(kernlab)
library(ranger)

# Source utility functions
source("utils_shiny.R")
source("scenarios.R")

# Define UI
ui <- dashboardPage(
  dashboardHeader(title = "Materials Engineering ML Platform"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Data Generation", tabName = "data_gen", icon = icon("database")),
      menuItem("Scenario Selection", tabName = "scenarios", icon = icon("list")),
      menuItem("Data Preprocessing", tabName = "preprocessing", icon = icon("cogs")),
      menuItem("Model Configuration", tabName = "models", icon = icon("brain")),
      menuItem("Training & Tuning", tabName = "training", icon = icon("play")),
      menuItem("Model Evaluation", tabName = "evaluation", icon = icon("chart-line")),
      menuItem("Model Comparison", tabName = "comparison", icon = icon("balance-scale")),
      menuItem("Deployment", tabName = "deployment", icon = icon("rocket")),
      menuItem("Model Management", tabName = "management", icon = icon("folder"))
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper, .right-side {
          background-color: #f4f4f4;
        }
      "))
    ),
    
    tabItems(
      # Data Generation Tab
      tabItem(tabName = "data_gen",
        fluidRow(
          box(
            title = "Data Generation Parameters", status = "primary", solidHeader = TRUE,
            width = 4,
            selectInput("data_type", "Material Property:",
                       choices = list(
                         "Thermal Conductivity" = "thermal",
                         "Electrical Conductivity" = "electrical", 
                         "Mechanical Strength" = "mechanical",
                         "Corrosion Resistance" = "corrosion",
                         "Fatigue Life" = "fatigue",
                         "Hardness" = "hardness",
                         "Elastic Modulus" = "elastic",
                         "Fracture Toughness" = "fracture",
                         "Creep Resistance" = "creep",
                         "Thermal Expansion" = "thermal_exp"
                       )),
            numericInput("n_samples", "Number of Samples:", value = 300, min = 50, max = 2000),
            numericInput("noise_level", "Noise Level:", value = 0.1, min = 0.01, max = 1.0, step = 0.01),
            numericInput("random_seed", "Random Seed:", value = 42, min = 1, max = 10000),
            actionButton("generate_data", "Generate Dataset", class = "btn-primary")
          ),
          
          box(
            title = "Dataset Preview", status = "info", solidHeader = TRUE,
            width = 8,
            DT::dataTableOutput("data_preview")
          )
        ),
        
        fluidRow(
          box(
            title = "Data Visualization", status = "success", solidHeader = TRUE,
            width = 12,
            plotlyOutput("data_plot", height = "400px")
          )
        )
      ),
      
      # Scenario Selection Tab
      tabItem(tabName = "scenarios",
        fluidRow(
          box(
            title = "Materials Engineering Scenarios", status = "primary", solidHeader = TRUE,
            width = 4,
            selectInput("scenario_category", "Scenario Category:",
                       choices = list(
                         "Alloy Development" = "alloy",
                         "Ceramic Processing" = "ceramic",
                         "Polymer Engineering" = "polymer",
                         "Composite Materials" = "composite",
                         "Nanomaterials" = "nano",
                         "Biomaterials" = "bio",
                         "Electronic Materials" = "electronic",
                         "Energy Materials" = "energy"
                       )),
            selectInput("specific_scenario", "Specific Scenario:",
                       choices = NULL),
            actionButton("load_scenario", "Load Scenario", class = "btn-success")
          ),
          
          box(
            title = "Scenario Description", status = "info", solidHeader = TRUE,
            width = 8,
            verbatimTextOutput("scenario_description"),
            br(),
            h4("Key Variables:"),
            verbatimTextOutput("scenario_variables"),
            br(),
            h4("Expected Relationships:"),
            verbatimTextOutput("scenario_relationships")
          )
        ),
        
        fluidRow(
          box(
            title = "Scenario Data", status = "warning", solidHeader = TRUE,
            width = 12,
            DT::dataTableOutput("scenario_data")
          )
        )
      ),
      
      # Data Preprocessing Tab
      tabItem(tabName = "preprocessing",
        fluidRow(
          box(
            title = "Data Splitting", status = "primary", solidHeader = TRUE,
            width = 6,
            numericInput("train_prop", "Training Proportion:", value = 0.7, min = 0.5, max = 0.9, step = 0.05),
            numericInput("val_prop", "Validation Proportion:", value = 0.15, min = 0.1, max = 0.3, step = 0.05),
            checkboxInput("stratified", "Stratified Sampling", value = FALSE),
            actionButton("split_data", "Split Data", class = "btn-primary")
          ),
          
          box(
            title = "Feature Engineering", status = "info", solidHeader = TRUE,
            width = 6,
            checkboxGroupInput("preprocessing_steps", "Preprocessing Steps:",
                              choices = list(
                                "Normalize/Scale" = "normalize",
                                "Center Variables" = "center",
                                "Remove Near-Zero Variance" = "nzv",
                                "Create Interactions" = "interactions",
                                "Polynomial Features" = "poly",
                                "PCA Transformation" = "pca",
                                "Log Transform" = "log",
                                "Box-Cox Transform" = "boxcox"
                              ),
                              selected = c("normalize", "center")),
            conditionalPanel(
              condition = "input.preprocessing_steps.includes('poly')",
              numericInput("poly_degree", "Polynomial Degree:", value = 2, min = 2, max = 4)
            ),
            conditionalPanel(
              condition = "input.preprocessing_steps.includes('pca')",
              numericInput("pca_threshold", "PCA Variance Threshold:", value = 0.95, min = 0.8, max = 0.99, step = 0.01)
            )
          )
        ),
        
        fluidRow(
          box(
            title = "Data Split Summary", status = "success", solidHeader = TRUE,
            width = 6,
            verbatimTextOutput("split_summary")
          ),
          
          box(
            title = "Recipe Summary", status = "warning", solidHeader = TRUE,
            width = 6,
            verbatimTextOutput("recipe_summary")
          )
        )
      ),
      
      # Model Configuration Tab
      tabItem(tabName = "models",
        fluidRow(
          box(
            title = "Model Selection", status = "primary", solidHeader = TRUE,
            width = 6,
            checkboxGroupInput("selected_models", "Select Models:",
                              choices = list(
                                "Linear Regression" = "linear_reg",
                                "Ridge Regression" = "ridge",
                                "Lasso Regression" = "lasso", 
                                "Elastic Net" = "elastic_net",
                                "Random Forest" = "random_forest",
                                "Support Vector Machine" = "svm",
                                "K-Nearest Neighbors" = "knn",
                                "Neural Network" = "nnet",
                                "Gradient Boosting" = "xgboost",
                                "Decision Tree" = "decision_tree"
                              ),
                              selected = c("linear_reg", "ridge", "random_forest")),
            actionButton("configure_models", "Configure Models", class = "btn-primary")
          ),
          
          box(
            title = "Hyperparameter Tuning", status = "info", solidHeader = TRUE,
            width = 6,
            selectInput("tuning_method", "Tuning Method:",
                       choices = list(
                         "Grid Search" = "grid",
                         "Random Search" = "random",
                         "Bayesian Optimization" = "bayes"
                       )),
            numericInput("cv_folds", "Cross-Validation Folds:", value = 5, min = 3, max = 10),
            numericInput("grid_size", "Grid Size:", value = 10, min = 5, max = 50),
            numericInput("n_iter", "Random/Bayes Iterations:", value = 20, min = 10, max = 100)
          )
        ),
        
        fluidRow(
          box(
            title = "Workflow Set Configuration", status = "success", solidHeader = TRUE,
            width = 12,
            verbatimTextOutput("workflow_summary"),
            br(),
            actionButton("create_workflows", "Create Workflow Set", class = "btn-success")
          )
        )
      ),
      
      # Training & Tuning Tab
      tabItem(tabName = "training",
        fluidRow(
          box(
            title = "Training Control", status = "primary", solidHeader = TRUE,
            width = 4,
            actionButton("start_training", "Start Training", class = "btn-success btn-lg"),
            br(), br(),
            checkboxInput("parallel_processing", "Enable Parallel Processing", value = TRUE),
            numericInput("n_cores", "Number of Cores:", value = 2, min = 1, max = 8),
            br(),
            progressBar(id = "training_progress", value = 0, title = "Training Progress")
          ),
          
          box(
            title = "Training Log", status = "info", solidHeader = TRUE,
            width = 8,
            verbatimTextOutput("training_log", placeholder = TRUE)
          )
        ),
        
        fluidRow(
          box(
            title = "Real-time Performance", status = "warning", solidHeader = TRUE,
            width = 12,
            plotlyOutput("training_progress_plot", height = "400px")
          )
        )
      ),
      
      # Model Evaluation Tab
      tabItem(tabName = "evaluation",
        fluidRow(
          box(
            title = "Model Selection", status = "primary", solidHeader = TRUE,
            width = 4,
            selectInput("eval_model", "Select Model for Evaluation:",
                       choices = NULL),
            br(),
            h4("Performance Metrics:"),
            verbatimTextOutput("model_metrics")
          ),
          
          box(
            title = "Prediction vs Actual", status = "info", solidHeader = TRUE,
            width = 8,
            plotlyOutput("pred_vs_actual_plot", height = "400px")
          )
        ),
        
        fluidRow(
          box(
            title = "Residual Analysis", status = "success", solidHeader = TRUE,
            width = 6,
            plotlyOutput("residual_plot", height = "350px")
          ),
          
          box(
            title = "Feature Importance", status = "warning", solidHeader = TRUE,
            width = 6,
            plotlyOutput("feature_importance_plot", height = "350px")
          )
        ),
        
        fluidRow(
          box(
            title = "Learning Curves", status = "danger", solidHeader = TRUE,
            width = 12,
            plotlyOutput("learning_curves_plot", height = "400px")
          )
        )
      ),
      
      # Model Comparison Tab
      tabItem(tabName = "comparison",
        fluidRow(
          box(
            title = "Comparison Metrics", status = "primary", solidHeader = TRUE,
            width = 4,
            selectInput("comparison_metric", "Primary Metric:",
                       choices = list(
                         "R-squared" = "rsq",
                         "RMSE" = "rmse",
                         "MAE" = "mae",
                         "MAPE" = "mape"
                       )),
            checkboxInput("show_confidence", "Show Confidence Intervals", value = TRUE),
            actionButton("update_comparison", "Update Comparison", class = "btn-primary")
          ),
          
          box(
            title = "Model Rankings", status = "info", solidHeader = TRUE,
            width = 8,
            DT::dataTableOutput("model_rankings")
          )
        ),
        
        fluidRow(
          box(
            title = "Performance Comparison", status = "success", solidHeader = TRUE,
            width = 6,
            plotlyOutput("model_comparison_plot", height = "400px")
          ),
          
          box(
            title = "Overfitting Analysis", status = "warning", solidHeader = TRUE,
            width = 6,
            plotlyOutput("overfitting_plot", height = "400px")
          )
        ),
        
        fluidRow(
          box(
            title = "Statistical Significance", status = "danger", solidHeader = TRUE,
            width = 12,
            DT::dataTableOutput("significance_tests")
          )
        )
      ),
      
      # Deployment Tab
      tabItem(tabName = "deployment",
        fluidRow(
          box(
            title = "Model Deployment", status = "primary", solidHeader = TRUE,
            width = 6,
            selectInput("deploy_model", "Select Model to Deploy:",
                       choices = NULL),
            textInput("model_name", "Model Name:", value = "materials_ml_model"),
            textInput("model_version", "Version:", value = "1.0.0"),
            textAreaInput("model_description", "Description:", 
                         value = "Materials engineering ML model for property prediction"),
            actionButton("deploy_model_btn", "Deploy Model", class = "btn-success")
          ),
          
          box(
            title = "Prediction Interface", status = "info", solidHeader = TRUE,
            width = 6,
            h4("Make Predictions:"),
            uiOutput("prediction_inputs"),
            br(),
            actionButton("make_prediction", "Predict", class = "btn-primary"),
            br(), br(),
            h4("Prediction Result:"),
            verbatimTextOutput("prediction_result")
          )
        ),
        
        fluidRow(
          box(
            title = "Model Export", status = "warning", solidHeader = TRUE,
            width = 12,
            h4("Export Options:"),
            checkboxGroupInput("export_formats", "Export Formats:",
                              choices = list(
                                "R RDS File" = "rds",
                                "PMML" = "pmml",
                                "ONNX" = "onnx",
                                "Model Report" = "report"
                              ),
                              selected = c("rds", "report")),
            actionButton("export_model", "Export Model", class = "btn-warning"),
            br(), br(),
            downloadButton("download_model", "Download Model Files", class = "btn-info")
          )
        )
      ),
      
      # Model Management Tab
      tabItem(tabName = "management",
        fluidRow(
          box(
            title = "Model Registry", status = "primary", solidHeader = TRUE,
            width = 8,
            DT::dataTableOutput("model_registry")
          ),
          
          box(
            title = "Model Actions", status = "info", solidHeader = TRUE,
            width = 4,
            selectInput("manage_model", "Select Model:",
                       choices = NULL),
            br(),
            actionButton("load_model", "Load Model", class = "btn-primary"),
            br(), br(),
            actionButton("archive_model", "Archive Model", class = "btn-warning"),
            br(), br(),
            actionButton("delete_model", "Delete Model", class = "btn-danger")
          )
        ),
        
        fluidRow(
          box(
            title = "Model Performance History", status = "success", solidHeader = TRUE,
            width = 12,
            plotlyOutput("performance_history_plot", height = "400px")
          )
        )
      )
    )
  )
)

# Define Server
server <- function(input, output, session) {
  # Reactive values to store data and models
  values <- reactiveValues(
    raw_data = NULL,
    processed_data = NULL,
    data_split = NULL,
    recipe = NULL,
    workflow_set = NULL,
    trained_models = NULL,
    selected_scenario = NULL,
    model_registry = data.frame()
  )
  
  # Data Generation
  observeEvent(input$generate_data, {
    values$raw_data <- generate_materials_data(
      data_type = input$data_type,
      n_samples = input$n_samples,
      noise_level = input$noise_level,
      random_seed = input$random_seed
    )
    
    showNotification("Dataset generated successfully!", type = "success")
  })
  
  output$data_preview <- DT::renderDataTable({
    req(values$raw_data)
    DT::datatable(values$raw_data, options = list(scrollX = TRUE))
  })
  
  output$data_plot <- renderPlotly({
    req(values$raw_data)
    create_data_visualization(values$raw_data, input$data_type)
  })
  
  # Scenario Selection
  observe({
    scenarios <- get_scenarios_by_category(input$scenario_category)
    updateSelectInput(session, "specific_scenario", 
                     choices = scenarios)
  })
  
  observeEvent(input$load_scenario, {
    values$selected_scenario <- load_scenario_data(
      input$scenario_category, 
      input$specific_scenario
    )
    values$raw_data <- values$selected_scenario$data
    
    showNotification("Scenario loaded successfully!", type = "success")
  })
  
  output$scenario_description <- renderText({
    req(values$selected_scenario)
    values$selected_scenario$description
  })
  
  output$scenario_variables <- renderText({
    req(values$selected_scenario)
    paste(values$selected_scenario$variables, collapse = "\n")
  })
  
  output$scenario_relationships <- renderText({
    req(values$selected_scenario)
    paste(values$selected_scenario$relationships, collapse = "\n")
  })
  
  output$scenario_data <- DT::renderDataTable({
    req(values$selected_scenario)
    DT::datatable(values$selected_scenario$data, options = list(scrollX = TRUE))
  })
  
  # Data Preprocessing
  observeEvent(input$split_data, {
    req(values$raw_data)
    
    values$data_split <- create_data_split(
      values$raw_data,
      train_prop = input$train_prop,
      val_prop = input$val_prop,
      stratified = input$stratified
    )
    
    showNotification("Data split completed!", type = "success")
  })
  
  observe({
    req(values$data_split, input$preprocessing_steps)
    
    values$recipe <- create_preprocessing_recipe(
      values$data_split,
      steps = input$preprocessing_steps,
      poly_degree = input$poly_degree,
      pca_threshold = input$pca_threshold
    )
  })
  
  output$split_summary <- renderText({
    req(values$data_split)
    create_split_summary(values$data_split)
  })
  
  output$recipe_summary <- renderText({
    req(values$recipe)
    capture.output(print(values$recipe))
  })
  
  # Model Configuration
  observeEvent(input$create_workflows, {
    req(values$recipe, input$selected_models)
    
    values$workflow_set <- create_workflow_set_models(
      recipe = values$recipe,
      models = input$selected_models
    )
    
    showNotification("Workflow set created!", type = "success")
  })
  
  output$workflow_summary <- renderText({
    req(values$workflow_set)
    capture.output(print(values$workflow_set))
  })
  
  # Training & Tuning
  observeEvent(input$start_training, {
    req(values$workflow_set, values$data_split)
    
    # Show progress
    updateProgressBar(session, "training_progress", value = 10, title = "Initializing...")
    
    # Start training in background
    values$trained_models <- train_workflow_set(
      workflow_set = values$workflow_set,
      data_split = values$data_split,
      cv_folds = input$cv_folds,
      tuning_method = input$tuning_method,
      grid_size = input$grid_size,
      n_iter = input$n_iter,
      parallel = input$parallel_processing,
      n_cores = input$n_cores,
      progress_callback = function(progress) {
        updateProgressBar(session, "training_progress", value = progress)
      }
    )
    
    updateProgressBar(session, "training_progress", value = 100, title = "Training Complete!")
    showNotification("Model training completed!", type = "success")
    
    # Update model choices for evaluation
    model_choices <- names(values$trained_models)
    updateSelectInput(session, "eval_model", choices = model_choices)
    updateSelectInput(session, "deploy_model", choices = model_choices)
    updateSelectInput(session, "manage_model", choices = model_choices)
  })
  
  output$training_log <- renderText({
    req(values$trained_models)
    "Training completed successfully. Check model evaluation for results."
  })
  
  output$training_progress_plot <- renderPlotly({
    req(values$trained_models)
    create_training_progress_plot(values$trained_models)
  })
  
  # Model Evaluation
  output$model_metrics <- renderText({
    req(values$trained_models, input$eval_model)
    get_model_metrics(values$trained_models, input$eval_model)
  })
  
  output$pred_vs_actual_plot <- renderPlotly({
    req(values$trained_models, input$eval_model, values$data_split)
    create_pred_vs_actual_plot(values$trained_models, input$eval_model, values$data_split)
  })
  
  output$residual_plot <- renderPlotly({
    req(values$trained_models, input$eval_model, values$data_split)
    create_residual_plot(values$trained_models, input$eval_model, values$data_split)
  })
  
  output$feature_importance_plot <- renderPlotly({
    req(values$trained_models, input$eval_model)
    create_feature_importance_plot(values$trained_models, input$eval_model)
  })
  
  output$learning_curves_plot <- renderPlotly({
    req(values$trained_models, input$eval_model, values$data_split)
    create_learning_curves_plot(values$trained_models, input$eval_model, values$data_split)
  })
  
  # Model Comparison
  output$model_rankings <- DT::renderDataTable({
    req(values$trained_models)
    create_model_rankings(values$trained_models, input$comparison_metric)
  })
  
  output$model_comparison_plot <- renderPlotly({
    req(values$trained_models)
    create_model_comparison_plot(values$trained_models, input$comparison_metric, input$show_confidence)
  })
  
  output$overfitting_plot <- renderPlotly({
    req(values$trained_models)
    create_overfitting_analysis_plot(values$trained_models)
  })
  
  output$significance_tests <- DT::renderDataTable({
    req(values$trained_models)
    perform_significance_tests(values$trained_models)
  })
  
  # Deployment
  output$prediction_inputs <- renderUI({
    req(values$raw_data)
    create_prediction_inputs(values$raw_data)
  })
  
  observeEvent(input$make_prediction, {
    req(values$trained_models, input$deploy_model)
    
    # Get input values and make prediction
    prediction <- make_model_prediction(
      values$trained_models, 
      input$deploy_model,
      get_prediction_input_values(input, values$raw_data)
    )
    
    output$prediction_result <- renderText({
      paste("Predicted value:", round(prediction, 3))
    })
  })
  
  observeEvent(input$deploy_model_btn, {
    req(values$trained_models, input$deploy_model)
    
    # Add to model registry
    new_model <- data.frame(
      name = input$model_name,
      version = input$model_version,
      model_type = input$deploy_model,
      description = input$model_description,
      deployed_date = Sys.Date(),
      status = "Active"
    )
    
    values$model_registry <- rbind(values$model_registry, new_model)
    
    showNotification("Model deployed successfully!", type = "success")
  })
  
  # Model Management
  output$model_registry <- DT::renderDataTable({
    values$model_registry
  })
  
  output$performance_history_plot <- renderPlotly({
    req(values$model_registry)
    create_performance_history_plot(values$model_registry)
  })
}

# Run the application
shinyApp(ui = ui, server = server)
