library(shiny)
library(shinythemes)
library(ggplot2)
library(corrplot)
library(randomForest)
library(class)

# Load dataset
red_wine <- red_wine <- read.csv2("./data/winequality-red.csv")


# UI
ui <- fluidPage(
  theme = shinytheme("flatly"), # Set a stylish theme
  titlePanel("Red Wine Quality Prediction Dashboard", windowTitle = "Wine Quality Predictor"),
  sidebarLayout(
    sidebarPanel(
      h4("Input Settings"), # Section title
      checkboxGroupInput("variables", "Select Variables for Prediction:",
                         choices = names(red_wine)[-ncol(red_wine)], 
                         selected = names(red_wine)[-ncol(red_wine)]),
      selectInput("model", "Select Model:", 
                  choices = c("KNN", "Random Forest")),
      numericInput("hyperparam", "Set Hyperparameter (e.g., k or ntrees):", value = 5),
      sliderInput("split_ratio", "Train/Test Split Ratio:", min = 0.1, max = 0.9, value = 0.7),
      numericInput("seed", "Set Seed for Reproducibility:", value = 42),
      actionButton("predict", "Run Prediction", class = "btn-primary"), # Styled button
      br(),
      tags$hr(), # Divider for better visuals
      tags$p("Adjust parameters and click 'Run Prediction' to view results.", 
             style = "color: #555; font-style: italic;")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Correlation", plotOutput("correlogram")), # Tabbed interface
        tabPanel("Predictions", plotOutput("barplot")),
        tabPanel("Accuracy", verbatimTextOutput("accuracy"))
      ),
      tags$br(), # Space between elements
      tags$hr(),
      tags$footer("Developed with Shiny", 
                  style = "text-align: center; font-size: 0.8em; color: #aaa;")
    )
  )
)

# Server
server <- function(input, output) {
  observeEvent(input$predict, {
    req(input$variables)
    
    # Prepare data with character-to-numeric conversion
    red_wine_processed <- red_wine
    red_wine_processed[] <- lapply(red_wine_processed, function(x) {
      if (is.character(x)) as.numeric(x) else x
    })
    
    X <- red_wine_processed[, input$variables, drop = FALSE]
    y <- red_wine_processed$quality
    set.seed(input$seed)
    train_index <- sample(seq_len(nrow(red_wine_processed)), size = input$split_ratio * nrow(red_wine_processed))
    X_train <- X[train_index, , drop = FALSE]
    y_train <- y[train_index]
    X_test <- X[-train_index, , drop = FALSE]
    y_test <- y[-train_index]
    
    # Train model
    if (input$model == "KNN") {
      y_pred <- knn(scale(X_train), scale(X_test), y_train, k = input$hyperparam)
    } else if (input$model == "Random Forest") {
      model <- randomForest(x = scale(X_train), y = as.factor(y_train), ntree = input$hyperparam)
      y_pred <- predict(model, scale(X_test))
    }
    
    # Accuracy
    accuracy <- mean(y_pred == y_test)
    
    # Correlogram
    output$correlogram <- renderPlot({
      corr <- cor(X)
      corrplot(corr, method = "color")
    })
    
    # Bar plot of predictions
    output$barplot <- renderPlot({
      pred_counts <- table(y_pred)
      ggplot(data.frame(Quality = names(pred_counts), Count = as.numeric(pred_counts)), aes(x = Quality, y = Count)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        labs(title = "Predicted Wine Quality Distribution", x = "Quality", y = "Count") +
        theme_minimal() # Clean barplot style
    })
    
    # Accuracy output
    output$accuracy <- renderText({
      paste("Model Accuracy:", round(accuracy, 2))
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)