#Caleb Brooks

library(tidyverse)
library(caret)
library(reshape2)

# Package for easy timing in R
library(tictoc)



# Demo of timer function --------------------------------------------------
# Run the next 5 lines at once
tic()
Sys.sleep(3)
timer_info <- toc()
runtime <- timer_info$toc - timer_info$tic
runtime



# Get data ----------------------------------------------------------------
# Accelerometer Biometric Competition Kaggle competition data
# https://www.kaggle.com/c/accelerometer-biometric-competition/data
train <- read_csv("~/Stat_495_git/PS08/train.csv")

# YOOGE!
dim(train)



# knn modeling ------------------------------------------------------------
model_formula <- as.formula(Device ~ X + Y + Z)

# Values to use:
n_values <- c(500000, 1000000, 3000000, 5000000)
k_values <- c(100, 500, 1000, 5000, 10000)

runtime_dataframe <- expand.grid(n_values, k_values) %>%
  as_tibble() %>%
  rename(n=Var1, k=Var2) %>%
  mutate(runtime = n*k)
runtime_dataframe


# Time knn here -----------------------------------------------------------
runtime <- data.frame('2'=c(0,0,0,0),'3'=c(0,0,0),'4'=c(0,0,0,0), '5'=c(0,0,0,0), '6'=c(0,0,0,0))
colnames(runtime)<-c('k=100','k=500','k=1000')
row.names(runtime)<-c('n=5e5','n=1e6','n=5e6')


for (n in 1:4){
  this_frame <- sample_n(train, n_values[n])
  for (k in 1:5){
    tic()
    model_formula <- as.formula(Device ~ X + Y + Z)
    model_knn <- caret::knn3(model_formula, data=this_frame, k = k_values[k])
    timer_info <- toc()
    runtime[n,k] <- (timer_info$toc - timer_info$tic)
  }
}




#sloppy reformatting to save runtime
runtime_melted <- melt(runtime) 
colnames(runtime_melted)[1] <- "k"
colnames(runtime_melted)[2] <- "runtime"
runtime_melted[1] <- c(100, 100, 100, 100, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 5000, 5000, 5000, 5000, 10000, 10000, 10000, 10000)
runtime_melted[,'n'] <- c(500000, 1000000, 3000000, 5000000, 500000, 1000000, 3000000, 5000000,500000, 1000000, 3000000, 5000000, 500000, 1000000, 3000000, 5000000, 500000, 1000000, 3000000, 5000000)
runtime_melted[,c(2,3)]<-runtime_melted[,c(3,2)]
colnames(runtime_melted)[c(2,3)]<-colnames(runtime_melted)[c(3,2)]
runtime_melted

# Plot your results ---------------------------------------------------------
# Think of creative ways to improve this barebones plot. Note: you don't have to
# necessarily use geom_point
runtime_plot <- ggplot(runtime_melted, aes(x=n, y=k, col=runtime)) +
  geom_point()

runtime_plot
ggsave(filename="caleb_brooks.png", width=16, height = 9)




# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:
# -n: number of points in training set
# Linear relationship (n) (assuming fixed k and d)
# -k: number of neighbors to consider
# linear relationship (assuming fixed n and d) (MUCH less significant coefficient than for n)
# -d: number of predictors used? In this case d is fixed at 3

