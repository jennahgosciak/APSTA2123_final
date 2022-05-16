data {
  int<lower = 0> N;
  int<lower = 0> N_movers;
  int<lower = 0> N_stayers;
  int<lower = 0> K; // number of predictors (1 in this example)
  matrix[N_movers, K + 1] X_movers_s;
  matrix[N_stayers, K + 1] X_stayers_s;
  matrix[N_movers,  K] X_movers;
  matrix[N_stayers, K] X_stayers;
  vector[N_movers]  y_movers;
  vector[N_stayers] y_stayers;
  int<lower = 0, upper = 1> prior_only;
  vector[5] m;
  vector<lower = 0>[5] scale;
  real<lower = 0> r;
}

parameters {
  real<lower = 0> sigma;
  real<lower = -1, upper = 1> rho;
  real alpha0;
  vector[K + 1] alpha1;
  real beta0;
  vector[K] beta1;
  real beta2;
}

transformed parameters {
  vector[N_movers]  mu_movers  = beta0 + X_movers*beta1 + beta2;
  vector[N_stayers] mu_stayers = beta0 + X_stayers*beta1;
  vector[N_movers] eta_movers = alpha0 + X_movers_s * alpha1 + 
      rho / sigma * (y_movers - mu_movers);
  vector[N_stayers] eta_stayers = alpha0 + X_stayers_s * alpha1 + 
      rho / sigma * (y_stayers - mu_stayers);
        
  real conditional_sd = sqrt(1 - square(rho));
}

model {
  if (!prior_only) {
    target += normal_lpdf(y_movers  | mu_movers, sigma);
    target += normal_lpdf(y_stayers | mu_stayers, sigma);
      
    target += normal_lcdf(0 | -eta_movers, conditional_sd);
    target += normal_lcdf(0 | eta_stayers, conditional_sd);
  }
  target += normal_lpdf(alpha0  | m[1], scale[1]);
  target += normal_lpdf(alpha1  | m[2], scale[2]);
  target += normal_lpdf(beta0   | m[3], scale[3]);
  target += normal_lpdf(beta1   | m[4], scale[4]);
  target += normal_lpdf(beta2   | m[5], scale[5]);
  target += exponential_lpdf(sigma | r);
  // implicit: rho ~ uniform(0,1)
}

generated quantities {
  vector[N] log_lik;
  {
    for (n in 1:N_movers) {
      log_lik[n] = normal_lpdf(y_movers[n] | mu_movers[n], sigma) + 
                       normal_lcdf(0 | -eta_movers[n], conditional_sd);
    }
    
    for (n in 1:N_stayers) {
      log_lik[N_movers + n] = normal_lpdf(y_stayers[n] | mu_stayers[n], sigma) + 
                       normal_lcdf(0 | -eta_stayers[n], conditional_sd);
    }
  }
}
