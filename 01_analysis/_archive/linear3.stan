#include quantile_functions.stan
data {
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of predictors
  matrix[N, K] X;   // matrix of predictors
  vector[N] y;      // outcomes
  int<lower = 0, upper = 1> prior_only;   // ignore data?
  vector[K + 2] m;                        // prior medians
  vector<lower = 0>[K + 2] scale;             // prior IQRs
}
parameters {
  vector[K] beta;
  real alpha;
  real<lower = 0> sigma;
}

model { // log likelihood, equivalent to target += normal_lpdf(y | alpha + X * beta, sigma)
  if (!prior_only) {
    vector[N] mu = alpha + X * beta;
    target += gamma_lpdf(y | shape, shape/exp(mu));
  }
  target += normal_lpdf(beta  | m[1],   scale[1]); // ^ important
  target += normal_lpdf(alpha | m[2:K + 1], scale[2:K + 1]);
  target += normal_lpdf(sigma | m[K + 2], scale[K + 2]); // actually half normal
}

generated quantities {
  vector[N] log_lik;
  vector[N] yrep;
  {
    vector[N] mu = alpha + X * beta;
    for (n in 1:N) {
      log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
      yrep[n] = normal_rng(mu[n], sigma);
    }
  }
}
