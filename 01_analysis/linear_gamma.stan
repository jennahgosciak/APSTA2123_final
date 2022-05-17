#include quantile_functions.stan
data {
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of predictors
  matrix[N, K] X;   // matrix of predictors
  vector<lower = 0>[N] y;      // outcomes
  int<lower = 0, upper = 1> prior_only;   // ignore data?
  vector[K + 1] m;                        // prior mean values
  vector<lower = 0>[K + 1] scale;         // prior scale values
  real r;
}
parameters {
  vector[K] beta;
  real alpha;
  real<lower = 0> shape; // shape cannot be less than 0
}

transformed parameters {
  vector[N] mu = alpha + X * beta;
}

model { // log likelihood
  if (!prior_only) {
    for (i in 1:N) target += gamma_lpdf(y[i] | shape, shape/exp(mu[i]));
  }
  target += normal_lpdf(alpha | m[1],   scale[1]); 
  target += normal_lpdf(beta | m[2:K + 1], scale[2:K + 1]);
  target += exponential_lpdf(shape | r); // exponential for shape param
}

generated quantities {
  vector[N] log_lik;
  vector[N] yrep;
  {
    for (n in 1:N) {
      log_lik[n] = gamma_lpdf(y[n] | shape, shape / exp(mu[n]));
      yrep[n] = gamma_rng(shape, shape / exp(mu[n]));
    }
  }
}
