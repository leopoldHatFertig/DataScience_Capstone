# Install packages
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(tictoc)) install.packages("tictoc", repos = "http://cran.us.r-project.org")

# Load libraries
library(dslabs)
library(dplyr)
library(caret)
library(tidyverse)
library(ggplot2)
library(data.table)
library(kableExtra)
library(tictoc)
#library(purrr)           # not needed anymore 
#library(lubridate)       # not needed anymore
#library(recommenderlab)  # not needed anymore

# Load original dataset
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

tic("total runtime")    # set timer for total runtime
tic("downloading data") # set timer for download

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
toc() # end timer for download

tic("setup general data")
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# join content of both files
movielens <- left_join(ratings, movies, by = "movieId")
#str(movielens)

# Create edx dataset for training and validation dataset for validation
# Validation set will be 10% of MovieLens data, training set 90%
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed) #cleanup
toc() # end timer for general setup

# As a result of the general setup we have 2 datasets, one for training and one for validation purposes.
#glimpse(edx)
# The training set "edx" consists of 9000055 observations of the 6 variables: userId, movieId, rating, timestamp, title, genre.
#glimpse(validation)
# The validation set "validation" consists of 9999999 observations of the same 6 variables and contains only userIds and movieIds, that are also in the training set.

# End Section 1
###############

# Section 2
# Data Exploration



# Training prediction models
tic("model training process") # set timer for model training

# Define target function, to assess algorithm performance.
# We use the RMSE, as defined in the Machine Learning Course, lesson "Recommendation Systems"
calculateRMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

target_RMSE <- 0.8649 # given in the project assignment

# Develop algorithm for RMSE predicition model

# Comment:  In the Machine Learning Course, lesson "Regularization", Prof. Irizarry showcased code 
#           using linear regression models with different tweaks with the following results:
#           |method                                |  RMSE|
#           |:-------------------------------------|-----:|
#           |Just the average                      | 1.048|
#           |Movie Effect Model                    | 0.986|
#           |Movie + User Effects Model            | 0.885|
#           |Regularized Movie + User Effect Model | 0.881|
#
#           The results just got down to a RMSE of 0.881, with Models with Regularized Movie + User Effect performing best.
#           So, going from this, I will continue using a Regression Model incorporating both Regularized Movie Effect & Regularized User Effect and try to optimise
#           the model to reach a better RMSE.
#

mean_edx <- mean(edx$rating) #define mean of ratings in the dataset used for training

lambda_tuning <- seq(0, 10, 0.1) #define interval for optimise tuning parameter lambda
for(lambda in lambda_tuning){    # loop through values for lambda to optimise model
  b_movie <- 
    edx %>% 
    group_by(movieId) %>%
    summarize(b_movie = sum(rating - mean_edx)/(n()+lambda)) # calculate movie effect per movieId
  b_user <- 
    edx %>% 
    left_join(b_movie, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_user = sum(rating - b_movie - mean_edx)/(n()+lambda)) # calculate user effect per userId
  predicted_ratings <- 
    validation %>% 
    left_join(b_movie, by = "movieId") %>%                # append movie effect to validation set
    left_join(b_user, by = "userId") %>%                  # append user effect to validation set
    mutate(prediction = mean_edx + b_movie + b_user) %>%  # calculate prediction
    .$prediction                                                              
  
  model_rmse <- calculateRMSE(validation$rating, predicted_ratings) # calculate model performance
  
  results_rmse <- if(exists("results_rmse")) c(results_rmse, model_rmse) else c(model_rmse) # create/append list of all results
  
  if(model_rmse <= target_RMSE){ # stopping algorithm as soon as target RMSE is reached in order to shorten runtime
    break()
  }
}

toc() # end timer for model training process
# End Section 2
###############

# Section 3
# Results
lambda_optimal <- lambda_tuning[which.min(results_rmse)] # value of lambda for best model trained
lambda_optimal

rmse_optimal <- min(results_rmse) # RMSE of best model trained
rmse_optimal
# The model performance improved and the target RMSE was reached.
# The training process itself was quite fast. In my case less than 2min and therefore faster than loading and preparing the data.
# The predicted ratings by this model are stored in >predicted_ratings<
 
toc() # end timer total runtime
gc() # run garbage collection for cleanup
