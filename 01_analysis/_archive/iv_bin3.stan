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
  real<lower = 0> r;
}

parameters {
  real<lower = 0> sigma;
  real<lower = -1, upper = 1> rho;
  real a0;
  vector[K + 1] a1;
  real b0;
  vector[K] b1;
  real b2;
}

transformed parameters {
  real conditional_sd = sqrt(1 - square(rho));
  real alpha0 = m[1] + scale[1] * a0;
  vector[K + 1] alpha1 = m[2] + scale[2] * a1;
  real beta0 = m[3] + scale[3] * b0;
  vector[K] beta1 = m[4] + scale[4] * b1;
  real beta2 = m[5] + scale[5] * b2;
  
  vector[N_child]  mu_child  = beta0 + X_child*beta1 + beta2;
  vector[N_nochild] mu_nochild = beta0 + X_nochild*beta1;
  vector[N_child] eta_child = alpha0 + X_child_s * alpha1 + 
      rho / sigma * (y_child - mu_child);
  vector[N_nochild] eta_nochild = alpha0 + X_nochild_s * alpha1 + 
      rho / sigma * (y_nochild - mu_nochild);
  
}
model {
  if (!prior_only) {
    target += normal_lpdf(y_child  | mu_child, sigma);
    target += normal_lpdf(y_nochild | mu_nochild, sigma);
      
    target += normal_lcdf(0 | eta_child, conditional_sd);
    target += normal_lcdf(0 | -eta_nochild, conditional_sd);
  }
  target += std_normal_lpdf(a0);
  target += std_normal_lpdf(a1);
  target += std_normal_lpdf(b0);
  target += std_normal_lpdf(b1);
  target += std_normal_lpdf(b2);
  target += exponential_lpdf(sigma | r);
  target += beta_lpdf(rho | 3, 3);
}

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
      moves_rep[n] = normal_rng(eta_child[n], conditional_sd) > 0;
      if (moves_rep[n] == 0) mu -= beta2;
      yrep[n] = normal_rng(mu, sigma);
    }
    
    for (n in 1:N_nochild) {
      log_lik[N_child + n] = normal_lpdf(y_nochild[n] | mu_nochild[n], sigma) + 
                       normal_lcdf(0 | -eta_nochild[n], conditional_sd);
      mu = mu_nochild[n];
      moves_rep[N_child + n] = normal_rng(eta_nochild[n], conditional_sd) > 0;
      if (moves_rep[N_child + n] == 1) mu += beta2;
      yrep[N_child + n] = normal_rng(mu, sigma);
    }
  }
}
