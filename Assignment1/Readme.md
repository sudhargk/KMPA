Assignment1 Plan

Some general notes

1. For Gaussian basis functions, we **do not** have to compute the variances or the covariance matrix. The variance sigma is an empirical input parameter. Refer to http://www.cs.columbia.edu/~jebara/4771/tutorials/regression.pdf
2. MATLAB has crossval with 'mse' parameter to cross validate using mean squared error
3. The pseudoinverse we are trying to compute is the Moore-Penrose psuedoinverse. Hence, we can use pinv for it. Refer to Bishop Ch3 pg142
