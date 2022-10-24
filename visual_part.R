library(reticulate)
library(dplyr)
library(lubridate)
library(data.table)

fx_rates <- fread('fx_rates_pln.csv', stringsAsFactors = F)
plot(fx_rates$fx, type = 'l', xlab = 'days', ylab = 'FX', frame = F)

reticulate::source_python('monte_carlo_project.py')

all_y <- sorted_df$expected_loss

hist(all_y, main = '', xlab = 'expected loss', ylab = 'frequency')

scenario_y <- final_result$expected_loss
hist(scenario_y, main = '', xlab = 'expected loss', ylab = 'frequency')

prob_of_positive_result
prob_not_loose_all

prices <- final_result$price
hist(prices, main = '', xlab = 'prices', ylab = 'frequency')
