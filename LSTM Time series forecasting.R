
library(keras)
library(tensorflow)


  
  Series = df$Value  # your time series 
  
  # transform data to stationarity
  diffed = diff(Series, differences = 1)
  
  
  # create a lagged dataset, i.e to be supervised learning
  
  lags <- function(x, k){
    
    lagged =  c(rep(NA, k), x[1:(length(x)-k)])
    DF = as.data.frame(cbind(lagged, x))
    colnames(DF) <- c( paste0('x-', k), 'x')
    DF[is.na(DF)] <- 0
    return(DF)
  }
  supervised = lags(diffed, k)
  
  
  ## split into train and test sets
  
  N = nrow(supervised)
  n = round(N *0.66, digits = 0)
  train = supervised[1:n, ]
  test  = supervised[(n+1):N,  ]
  
  
  ## scale data
  normalize <- function(train, test, feature_range = c(0, 1)) {
    x = train
    fr_min = feature_range[1]
    fr_max = feature_range[2]
    std_train = ((x - min(x) ) / (max(x) - min(x)  ))
    std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
    
    scaled_train = std_train *(fr_max -fr_min) + fr_min
    scaled_test = std_test *(fr_max -fr_min) + fr_min
    
    return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
    
  }
  
  
  ## inverse-transform
  inverter = function(scaled, scaler, feature_range = c(0, 1)){
    min = scaler[1]
    max = scaler[2]
    n = length(scaled)
    mins = feature_range[1]
    maxs = feature_range[2]
    inverted_dfs = numeric(n)
    
    for( i in 1:n){
      X = (scaled[i]- mins)/(maxs - mins)
      rawValues = X *(max - min) + min
      inverted_dfs[i] <- rawValues
    }
    return(inverted_dfs)
  }
  
  
  Scaled = normalize(train, test, c(-1, 1))
  
  y_train = Scaled$scaled_train[, 2]
  x_train = Scaled$scaled_train[, 1]
  
  y_test = Scaled$scaled_test[, 2]
  x_test = Scaled$scaled_test[, 1]
  
  ## fit the model
  
  dim(x_train) <- c(length(x_train), 1, 1)
  dim(x_train)
  X_shape2 = dim(x_train)[2]
  X_shape3 = dim(x_train)[3]
  batch_size = 1
  units = 1
  
  model <- keras_model_sequential() 
  model%>%
    layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
    layer_dense(units = 1)
  
  
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam( lr= 0.02 , decay = 1e-6 ),  
    metrics = c('accuracy')
  )
  
  
  
  summary(model)
  
  nb_epoch = Epochs   
  for(i in 1:nb_epoch ){
    model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
    model %>% reset_states()
  }
  
  
  L = length(x_test)
  dim(x_test) = c(length(x_test), 1, 1)
  
  scaler = Scaled$scaler

  predictions = numeric(L)
  for(i in 1:L){
    X = x_test[i , , ]
    dim(X) = c(1,1,1)
    # forecast
    yhat = model %>% predict(X, batch_size=batch_size)
    
    # invert scaling
    yhat = inverter(yhat, scaler,  c(-1, 1))
    
    # invert differencing
    yhat  = yhat + Series[(n+i)] 
    
    # save prediction
    predictions[i] <- yhat
  }
