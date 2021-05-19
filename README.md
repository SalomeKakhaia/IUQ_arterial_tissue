### Coefficient sampling
Force coefficients are sampled by Latin hypercube sampling (LHS) within the range of zero to upper value, obtained by approximating coefficients of the polynomial function based on the order of the coefficient. The parameters space is then created and sets of parameters are sampled using LHS for indices of the parameters space. Coefficients are updated in the config file of ISR3D uniaxial strain tests and corresponding AB model responses are collected.

### Gaussian process surrogate model
The Gaussian process regression model is used as a surrogate modelling technique to represent a mapping between input strain and force coefficients values and output stress results, such that generating accurate predictions for yet unobserved parameters is easily attainable. The matern 5/2 kernel is used and all the parameters are constrained to be positive.

### Sensitivity analysis

### Inverse Uncertainty Quantification
Inverse uncertainty quantification is performed using the Bayesian Calibration method. The model responses are simulated by the pre-trained GP regression model predictions. For the prior information, the uninformative uniform distribution of coefficients is used, where the lower bound of uniform distribution is set to 0 and the upper bound is obtained by approximation of coefficients of polynomial function based on their order.
When it comes to the uncertainty, the modelling error term is considered, which is normally distributed with zero-mean and variance $sigma$. Then, log-likelihood is calculated where the likelihood is modelled as a joint Gaussian distribution, whose mean is calibration data given GP model response for the prior uncertain parameters and the variance is $sigma$. Parameters used for modelling uncertainties are estimated along with the rest of the calibration parameters in the IUQ process. Finally, the posterior distributions of parameters are sampled.
The calibration process is then performed iteratively, in order to update our beliefs about the calibration parameters, obtained posterior distributions are used as prior distributions for the next inference and new posteriors are produced. This process of calibration is then iterated until posterior distributions converge. For reusing posteriors, first, Gaussian Kernel density estimation is used to estimate the PDF of a random variable in a non-parametric way. Then in order to sample from obtained PDFs, a linear interpolation of PDF was evaluated on evenly distributed points on the extended domain of posterior samples, so that non sampled values do not have probabilities of 0, but some small values. Sequential Monte Carlo sampler is used as a sampling technique.
