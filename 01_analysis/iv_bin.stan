#include quantile_functions.stan
data {
  int<lower = 0> N;
  int<lower = 0> N_child;
  int<lower = 0> N_nochild;
  int<lower = 0> K; // number of predictors
  matrix[N_child, K + 1] X_child_s;
  matrix[N_nochild, K + 1] X_nochild_s;
  matrix[N_child,  K] X_child;
  matrix[N_nochild, K] X_nochild;
  vector[N_child]  y_child;
  vector[N_nochild] y_nochild;
  int<lower = 0, upper = 1> prior_only;
  vector[5] m;
  vector<lower = 0>[5] scale;
  vector[4] r;
}

parameters {
  real<lower = 0> sigma;
  real<lower = 0, upper = 1> p; // CDF for rho
  real alpha0;
  vector[K + 1] alpha1;
  real beta0;
  vector[K] beta1;
  real beta2;
}

transformed parameters {
  real<lower = -1, upper = 1> rho = gld_qf(p, r[1], r[2], r[3], r[4]);
  vector[N_child]  mu_child  = beta0 + X_child*beta1 + beta2;
  vector[N_nochild] mu_nochild = beta0 + X_nochild*beta1;
  vector[N_child] eta_child = alpha0 + X_child_s * alpha1 + 
      rho / sigma * (y_child - mu_child);
  vector[N_nochild] eta_nochild = alpha0 + X_nochild_s * alpha1 + 
      rho / sigma * (y_nochild - mu_nochild);
        
  real conditional_sd = sqrt(1 - square(rho));
}

model {
  if (!prior_only) {
    target += normal_lpdf(y_child  | mu_child, sigma);
    target += normal_lpdf(y_nochild | mu_nochild, sigma);
      
    target += normal_lcdf(0 | eta_child, conditional_sd);
    target += normal_lcdf(0 | -eta_nochild, conditional_sd);
  }
  target += normal_lpdf(alpha0  | m[1], scale[1]);
  target += normal_lpdf(alpha1  | m[2], scale[2]);
  target += normal_lpdf(beta0   | m[3], scale[3]);
  target += normal_lpdf(beta1   | m[4], scale[4]);
  target += normal_lpdf(beta2   | m[5], scale[5]);
  target += gamma_lpdf(sigma | 2, 2);
} // implicit: p ~ uniform(0, 1)

generated quantities {
  vector[N] log_lik;
  int moves_rep[N];
  vector[N] yrep;
  real mu;
  {
    for (n in 1:N_child) {
      log_lik[n] = normal_lpdf(y_child[n] | mu_child[n], sigma) + 
                       normal_lcdf(0 | eta_child[n], conditional_sd);
      mu = mu_child[n];
      
      /* intermediate outcome: if they have an additional child */
      moves_rep[n] = normal_rng(eta_child[n], conditional_sd) > 0;
      if (moves_rep[n] == 0) mu -= beta2; // subtract beta if not
      yrep[n] = normal_rng(mu, sigma); // sample outcome from new normal distribution
    }
    
    for (n in 1:N_nochild) {
      log_lik[N_child + n] = normal_lpdf(y_nochild[n] | mu_nochild[n], sigma) + 
                       normal_lcdf(0 | -eta_nochild[n], conditional_sd);
      mu = mu_nochild[n];
      
      /* intermediate outcome: if they have an additional child */
      moves_rep[N_child + n] = normal_rng(eta_nochild[n], conditional_sd) > 0;
      if (moves_rep[N_child + n] == 1) mu += beta2; // add beta if yes
      yrep[N_child + n] = normal_rng(mu, sigma); // sample outcome from new normal distribution
    }
  }
}
