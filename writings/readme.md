# Notes for Ryan Cloutier's odd-ratio submission based on cross-validation

* Conversely to most (all?) other submitted techniques, cross-validation (CV) is not used to estimate the marginalized likelihood, instead it is used to estimate the predictive power of each model on previously unseen RV data


## Leave-one-out CV
* The dataset of N RV measurements is split into N unique train/test sets of N-1 and 1 RV measurement respectively. Each training set is equal to the full RV dataset less one measurement which itself constitutes the testing set
* On each training set, the parameters of each planet model are optimized
* The lnlikelihood of measuring the unseen testing datum is then computed for each model using the optimized model parameters
* The sum of all N lnlikelihoods for each planet model are can then be compared (similarly to an odds ratio) to conclude which planet model is best suited to forecasting unseen measurements
* CV is a commonly used to avoid over-fitting of data as highly comples models can often be fine-tuned to provide a high likelihood whilst providing poor predictive power due to the finely-tuned parameters which do not necessarily generalize to unseen data (i.e. future observations)

## Time-Series Cross-Validation:
* Leave-one-out CV is applicable when measurements in the input dataset, which need not be a time-series, are uncorrelated. This is certainly not the case in an RV time-series whose adjacent points are highly correlated due to the presence of a periodic planetary signals and/or correlated noise arising from stellar activity
* When dealing with time-series, the removal of a single measurement does not remove all the information content associated with that dataum due to correlations within the data
* As such we modify our method of train/test splitting of the data according to https://projecteuclid.org/euclid.ssu/1268143839 as described below
  * y_1