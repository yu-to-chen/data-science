# Sample R code for "Bayesian Decision Theory Made Ridiculously Simple."
# http://www.statsathome.com/2017/10/12/bayesian-decision-theory-made-ridiculously-simple/
library(tidyverse)
library(brms)

d <- data.frame(sold = c(1, 1, 0, 0, 1, 1), 
           scratched = c(0, 0, 1, 0, 0, 0), 
           year = c(2014, 2015, 2010, 2014, 2015, 2016), 
           price = c(50, 70, 40, 100, 90, 100))
print(d)

# brms: An R Package for Bayesian Multilevel Models Using Stan.
# https://www.jstatsoft.org/article/view/v080i01
fit <- brm(sold ~ scratched + year + price, data = d, family = bernoulli(link = "logit"),
           prior = c(set_prior("normal(-1,1)", class="b", coef="scratched"),
                     set_prior("normal(1, 1)", class="b", coef="year"),
                     set_prior("normal(-2, 1)", class="b", coef="price")),
           silent=TRUE,
           refresh = -1)

print(summary(fit))

loss <- function(price){
  # First Create the input data to predict with out model - we want to predict whether or not our phone will sell
  our.phone <- data.frame(scratched=0, year=2014, price=price)

  # Next, for each posterior sample from out model, predict whether or not our phone would sell at the given price. This will give a vector of 0's and 1's, did the phone sell in each posterior sample. Think of each posterior sample as a simulation.
  pp <- posterior_predict(fit, newdata=our.phone)

  # Next calculate the expected return for each of these posterior simulations
  mean(pp*price)
}

# https://www.rdocumentation.org/packages/stats/versions/3.6.1/topics/optim
# 50 is the initial parameter value.
# Nelder-Mead method is the default, which works reasonably well for non-differentiable functions.
(op <- optim(50, function(x) -loss(x)))

x <- 10:110 # Listing prices to evaluate
l <- sapply(x, loss)
plot(x, l, xlab = "Listing Price", ylab = "Expected Return")

