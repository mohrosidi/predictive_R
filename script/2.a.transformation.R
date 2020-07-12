# Log transform a value
y <- log(10)

# Undo log-transformation
exp(y)


# Box Cox transform a value
y <- forecast::BoxCox(10, lambda)

# Inverse Box Cox function
inv_box_cox <- function(x, lambda) {
  # for Box-Cox, lambda = 0 --> log transform
  if (lambda == 0) exp(x) else (lambda*x + 1)^(1/lambda) 
}

# Undo Box Cox-transformation
inv_box_cox(y, lambda)
