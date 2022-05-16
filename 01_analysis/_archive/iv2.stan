data {              // saved as "iv.stan"
  int<lower = 0> N; // number of observations
  vector[N] q;       // quarter
  vector[N] s;                          // schooling
  vector[N] y;                          // outcome
  int<lower = 0, upper = 1> prior_only;
  vector[3] m;
  vector<lower = 0>[3] scale;
  real<lower = 0> r;
}
parameters {
  vector[4] lambda;
  vector<lower = 0>[2] sigma;
  real<lower = -1, upper = 1> rho;
  real alpha;
  real beta0;
  real beta1;
  real beta2;
}
model {
  if (!prior_only) {
    vector[N] s_ = lambda[q];
    vector[N] y_ = beta0 + beta1 * s + beta2 * q;
    target += normal_lpdf(s | s_, sigma[1]);
    target += normal_lpdf(y | y_, 
                          sigma[2] * sqrt(1 - square(rho)));
  }
  target += normal_lpdf(lambda | m[1], scale[1]);
  target += normal_lpdf(alpha  | m[2], scale[2]);
  target += normal_lpdf(beta   | m[3], scale[3]);
  target += exponential_lpdf(sigma | r);
  // implicit: rho ~ uniform(0,1)
}
