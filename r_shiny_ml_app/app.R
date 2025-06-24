# Comprehensive Materials Engineering Machine Learning Shiny Application
# Using tidymodels ecosystem for end-to-end ML workflows

library(shiny)
library(shinydashboard)
library(DT)
# library(plotly) # Optional dependency - commented out for compatibility
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
            plotOutput("data_plot", height = "400px")
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
            div(id = "training_progress", 
                style = "margin-top: 10px;",
                h5("Training Progress"),
                div(class = "progress", 
                    div(class = "progress-bar", role = "progressbar", 
                        style = "width: 0%", id = "progress_bar")))
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
            plotOutput("training_progress_plot", height = "400px")
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
            plotOutput("pred_vs_actual_plot", height = "400px")
          )
        ),
        
        fluidRow(
          box(
            title = "Residual Analysis", status = "success", solidHeader = TRUE,
            width = 6,
            plotOutput("residual_plot", height = "350px")
          ),
          
          box(
            title = "Feature Importance", status = "warning", solidHeader = TRUE,
            width = 6,
            plotOutput("feature_importance_plot", height = "350px")
          )
        ),
        
        fluidRow(
          box(
            title = "Learning Curves", status = "danger", solidHeader = TRUE,
            width = 12,
            plotOutput("learning_curves_plot", height = "400px")
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
            plotOutput("model_comparison_plot", height = "400px")
          ),
          
          box(
            title = "Overfitting Analysis", status = "warning", solidHeader = TRUE,
            width = 6,
            plotOutput("overfitting_plot", height = "400px")
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
            plotOutput("performance_history_plot", height = "400px")
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
    
    showNotification("Dataset generated successfully!", type = "message")
  })
  
  output$data_preview <- DT::renderDataTable({
    req(values$raw_data)
    DT::datatable(values$raw_data, options = list(scrollX = TRUE))
  })
  
  output$data_plot <- renderPlot({
    req(values$raw_data)
    create_data_visualization(values$raw_data, input$data_type)
  })
  
  # Scenario Selection
  observeEvent(input$scenario_category, {
    req(input$scenario_category)
    scenarios <- get_scenarios_by_category(input$scenario_category)
    updateSelectInput(session, "specific_scenario", 
                     choices = scenarios)
  })
  
  observeEvent(input$load_scenario, {
    req(input$scenario_category, input$specific_scenario)
    values$selected_scenario <- load_scenario_data(
      input$scenario_category, 
      input$specific_scenario
    )
    values$raw_data <- values$selected_scenario$data
    
    showNotification("Scenario loaded successfully!", type = "message")
  })
  
  observeEvent(input$generate_data, {
    req(input$category, input$scenario)
    
    withProgress(message = 'Generating data...', value = 0, {
      incProgress(0.3, detail = "Creating scenario data...")
      
      if (grepl("smiles", input$scenario) && !is.null(input$smiles_input) && input$smiles_input != "") {
        smiles_list <- strsplit(input$smiles_input, "\n")[[1]]
        smiles_list <- smiles_list[smiles_list != ""]
        
        if (input$calculate_descriptors) {
          incProgress(0.5, detail = "Calculating molecular descriptors...")
          descriptors <- parse_smiles_and_calculate_descriptors(smiles_list)
        }
        
        if (input$generate_fingerprints) {
          incProgress(0.7, detail = "Generating molecular fingerprints...")
          fingerprints <- generate_molecular_fingerprints(smiles_list)
        }
        
        data <- generate_scenario_data(input$category, input$scenario, length(smiles_list))
        if (exists("descriptors")) {
          descriptor_df <- do.call(rbind, lapply(descriptors, as.data.frame))
          data <- cbind(data, descriptor_df[, !names(descriptor_df) %in% names(data)])
        }
      } else {
        data <- generate_scenario_data(input$category, input$scenario, 200)
      }
      
      values$raw_data <- data
      
      incProgress(0.9, detail = "Processing data...")
      values$processed_data <- data
      
      if (grepl("transfer_learning", input$scenario)) {
        values$transfer_learning_mode <- input$transfer_learning_mode
        values$learning_rate <- input$learning_rate
      }
      
      incProgress(1, detail = "Complete!")
    })
    
    showNotification("Data generated successfully!", type = "success")
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
    
    showNotification("Data split completed!", type = "message")
  })
  
  observeEvent(list(values$data_split, input$preprocessing_steps), {
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
    
    showNotification("Workflow set created!", type = "message")
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
    
    if (!is.null(values$transfer_learning_mode) && values$transfer_learning_mode != "standard") {
      updateProgressBar(session, "training_progress", value = 30, title = "Setting up transfer learning...")
      
      if (values$transfer_learning_mode == "pretrained" && !is.null(input$base_model_upload)) {
        base_model_path <- input$base_model_upload$datapath
        base_model <- create_transfer_learning_workflow(base_model_path)
      } else {
        base_data <- values$raw_data[1:(nrow(values$raw_data) * 0.6), ]
        target_col <- names(values$raw_data)[ncol(values$raw_data)]
        base_model <- pretrain_base_model(base_data, target_col, "linear")
      }
      
      if (values$transfer_learning_mode == "finetune" && !is.null(base_model)) {
        updateProgressBar(session, "training_progress", value = 50, title = "Fine-tuning model...")
        target_data <- training(values$data_split)
        target_col <- names(target_data)[ncol(target_data)]
        finetuned_model <- finetune_model(base_model, target_data, target_col, values$learning_rate)
        
        values$base_model <- base_model
        values$finetuned_model <- finetuned_model
        
        updateProgressBar(session, "training_progress", value = 80, title = "Comparing models...")
        comparison <- create_transfer_learning_comparison(
          base_model, finetuned_model, testing(values$data_split), target_col
        )
        values$transfer_comparison <- comparison
      }
    } else {
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
      
      # Update model choices for evaluation
      model_choices <- names(values$trained_models)
      updateSelectInput(session, "eval_model", choices = model_choices)
      updateSelectInput(session, "deploy_model", choices = model_choices)
      updateSelectInput(session, "manage_model", choices = model_choices)
    }
    
    updateProgressBar(session, "training_progress", value = 100, title = "Training Complete!")
    showNotification("Model training completed!", type = "message")
  })
  
  output$training_log <- renderText({
    req(values$trained_models)
    "Training completed successfully. Check model evaluation for results."
  })
  
  output$training_progress_plot <- renderPlot({
    req(values$trained_models)
    create_training_progress_plot(values$trained_models)
  })
  
  # Model Evaluation
  output$model_metrics <- renderText({
    req(values$trained_models, input$eval_model)
    get_model_metrics(values$trained_models, input$eval_model)
  })
  
  output$pred_vs_actual_plot <- renderPlot({
    req(values$trained_models, input$eval_model, values$data_split)
    create_pred_vs_actual_plot(values$trained_models, input$eval_model, values$data_split)
  })
  
  output$residual_plot <- renderPlot({
    req(values$trained_models, input$eval_model, values$data_split)
    create_residual_plot(values$trained_models, input$eval_model, values$data_split)
  })
  
  output$feature_importance_plot <- renderPlot({
    req(values$trained_models, input$eval_model)
    create_feature_importance_plot(values$trained_models, input$eval_model)
  })
  
  output$learning_curves_plot <- renderPlot({
    req(values$trained_models, input$eval_model, values$data_split)
    create_learning_curves_plot(values$trained_models, input$eval_model, values$data_split)
  })
  
  # Model Comparison
  output$model_rankings <- DT::renderDataTable({
    req(values$trained_models)
    create_model_rankings(values$trained_models, input$comparison_metric)
  })
  
  output$model_comparison_plot <- renderPlot({
    if (!is.null(values$transfer_comparison)) {
      library(ggplot2)
      library(tidyr)
      
      comparison_long <- values$transfer_comparison %>%
        pivot_longer(cols = c(rmse, rsq, mae), names_to = "metric", values_to = "value")
      
      ggplot(comparison_long, aes(x = model, y = value, fill = model)) +
        geom_col() +
        facet_wrap(~metric, scales = "free_y") +
        labs(title = "Transfer Learning Model Comparison",
             x = "Model Type", y = "Performance") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    } else if (!is.null(values$trained_models)) {
      create_model_comparison_plot(values$trained_models, input$comparison_metric, input$show_confidence)
    } else {
      plot(1, type = "n", main = "No model results available")
    }
  })
  
  output$overfitting_plot <- renderPlot({
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
    
    showNotification("Model deployed successfully!", type = "message")
  })
  
  # Model Management
  output$model_registry <- DT::renderDataTable({
    values$model_registry
  })
  
  output$performance_history_plot <- renderPlot({
    req(values$model_registry)
    create_performance_history_plot(values$model_registry)
  })
  
  # Download Handler for Tar Archives
  output$download_model <- downloadHandler(
    filename = function() {
      paste0("materials_ml_models_", Sys.Date(), ".tar.gz")
    },
    content = function(file) {
      # Create temporary directory for files to archive
      temp_dir <- tempdir()
      archive_dir <- file.path(temp_dir, "ml_models_archive")
      dir.create(archive_dir, recursive = TRUE, showWarnings = FALSE)
      
      # Create subdirectories
      models_dir <- file.path(archive_dir, "models")
      data_dir <- file.path(archive_dir, "data")
      reports_dir <- file.path(archive_dir, "reports")
      dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
      dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
      dir.create(reports_dir, recursive = TRUE, showWarnings = FALSE)
      
      # Save trained models if available
      if (!is.null(values$trained_models)) {
        tryCatch({
          # Save workflow set as RDS
          saveRDS(values$trained_models, file.path(models_dir, "trained_models.rds"))
          
          # Save individual model files based on selected export formats
          if ("rds" %in% input$export_formats) {
            saveRDS(values$trained_models, file.path(models_dir, "models_rds_format.rds"))
          }
          
          # Create model summary report
          model_summary <- data.frame(
            model_id = names(values$trained_models$result),
            model_type = sapply(values$trained_models$result, function(x) class(x)[1]),
            created_date = Sys.time(),
            stringsAsFactors = FALSE
          )
          write.csv(model_summary, file.path(reports_dir, "model_summary.csv"), row.names = FALSE)
        }, error = function(e) {
          warning("Error saving models: ", e$message)
        })
      }
      
      # Save datasets if available
      if (!is.null(values$raw_data)) {
        tryCatch({
          write.csv(values$raw_data, file.path(data_dir, "raw_data.csv"), row.names = FALSE)
        }, error = function(e) {
          warning("Error saving raw data: ", e$message)
        })
      }
      
      if (!is.null(values$data_split)) {
        tryCatch({
          # Save training data
          train_data <- training(values$data_split)
          write.csv(train_data, file.path(data_dir, "training_data.csv"), row.names = FALSE)
          
          # Save testing data
          test_data <- testing(values$data_split)
          write.csv(test_data, file.path(data_dir, "testing_data.csv"), row.names = FALSE)
        }, error = function(e) {
          warning("Error saving split data: ", e$message)
        })
      }
      
      # Save scenario information if available
      if (!is.null(values$current_scenario)) {
        tryCatch({
          scenario_info <- list(
            scenario = values$current_scenario,
            category = input$scenario_category,
            generated_date = Sys.time()
          )
          saveRDS(scenario_info, file.path(reports_dir, "scenario_info.rds"))
        }, error = function(e) {
          warning("Error saving scenario info: ", e$message)
        })
      }
      
      # Save model registry if available
      if (!is.null(values$model_registry) && nrow(values$model_registry) > 0) {
        tryCatch({
          write.csv(values$model_registry, file.path(reports_dir, "model_registry.csv"), row.names = FALSE)
        }, error = function(e) {
          warning("Error saving model registry: ", e$message)
        })
      }
      
      # Create README file for the archive
      readme_content <- paste(
        "Materials Engineering ML Models Archive",
        "=====================================",
        "",
        paste("Generated on:", Sys.time()),
        paste("Archive contains:"),
        "- models/: Trained machine learning models",
        "- data/: Training and testing datasets", 
        "- reports/: Model summaries and metadata",
        "",
        "Files included:",
        paste("- Models:", ifelse(!is.null(values$trained_models), "Yes", "No")),
        paste("- Raw Data:", ifelse(!is.null(values$raw_data), "Yes", "No")),
        paste("- Split Data:", ifelse(!is.null(values$data_split), "Yes", "No")),
        paste("- Scenario Info:", ifelse(!is.null(values$current_scenario), "Yes", "No")),
        paste("- Model Registry:", ifelse(!is.null(values$model_registry) && nrow(values$model_registry) > 0, "Yes", "No")),
        "",
        "This archive was created by the Materials Engineering ML Platform",
        "using the tidymodels ecosystem in R.",
        sep = "\n"
      )
      writeLines(readme_content, file.path(archive_dir, "README.txt"))
      
      # Create the tar.gz archive
      tryCatch({
        # Change to temp directory to create relative paths in archive
        old_wd <- getwd()
        setwd(temp_dir)
        
        # Create tar.gz archive
        tar(file, files = basename(archive_dir), compression = "gzip")
        
        # Restore working directory
        setwd(old_wd)
        
        # Clean up temporary directory
        unlink(archive_dir, recursive = TRUE)
        
      }, error = function(e) {
        setwd(old_wd)
        stop("Error creating tar archive: ", e$message)
      })
    },
    contentType = "application/gzip"
  )
}

# Run the application
shinyApp(ui = ui, server = server)
