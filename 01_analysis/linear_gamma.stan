#include quantile_functions.stan
data {
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of predictors
  matrix[N, K] X;   // matrix of predictors
  vector[N] y;      // outcomes
  int<lower = 0, upper = 1> prior_only;   // ignore data?
  vector[K + 2] m;                        // prior medians
  vector<lower = 0>[K + 2] scale;             // prior IQRs
  real r;
}
parameters {
  vector[K] beta;
  real alpha;
  real<lower = 0> shape;
}

model { // log likelihood, equivalent to target += normal_lpdf(y | alpha + X * beta, sigma)
  if (!prior_only) {
    vector[N] mu = alpha + X * beta;
    for (i in 1:N) target += gamma_lpdf(y | shape, shape/exp(mu[i]));
  }
  target += normal_lpdf(beta  | m[1],   scale[1]); // ^ important
  target += normal_lpdf(alpha | m[2:K + 1], scale[2:K + 1]);
  target += exponential_lpdf(shape | r); // exponential
}

generated quantities {
  vector[N] log_lik;
  vector[N] yrep;
  {
    vector[N] mu = alpha + X * beta;
    for (n in 1:N) {
      log_lik[n] = gamma_lpdf(y[n] | shape, shape / exp(mu[n]));
      yrep[n] = gamma_rng(shape, shape / exp(mu[n]));
    }
  }
}
