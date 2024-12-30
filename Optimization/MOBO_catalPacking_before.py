import os
import torch
import pandas as pd
from torch.distributions import LowRankMultivariateNormal


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
MOBO_TEST = os.environ.get("MOBO_TEST")


from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples


def generate_initial_data():
    # generate training data
    expData = pd.read_csv('data.csv')
    train_x = torch.tensor(expData.values[:,0:3]).to(**tkwargs)
    train_obj = torch.tensor(expData.values[:,3:5]).to(**tkwargs)
    return train_x, train_obj


def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex


expData = pd.read_csv('data.csv')
train_x = torch.tensor(expData.values[:, 0:3]).to(**tkwargs)


standard_bounds = torch.zeros(3, 3, **tkwargs)
standard_bounds[1] = 1


# bounds = pd.read_csv('bounds.csv')
# bounds = torch.tensor(bounds.values).to(**tkwargs)


BATCH_SIZE = 4
NUM_RESTARTS = 1000
RAW_SAMPLES = 5000


def optimize_qehvi_and_get_observation(model, train_obj, sampler):
    partitioning = NondominatedPartitioning(ref_point=torch.tensor([0,0]).to(**tkwargs), Y=train_obj)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=[0,0], # use known reference point
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit":20, "maxiter":1000},
        sequential=True,
    )
    #observe new values
    #new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    #new_obj = problem(new_x)
    #print(new_x)
    #return new_x, new_obj
    return candidates.detach()



from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

import time
import warnings


warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

N_TRIALS = 1
N_BATCH = 25 # number of iterations
MC_SAMPLES = 1024

hvs_qehvi_all = []

hv = Hypervolume(ref_point=torch.tensor([0,0]).to(**tkwargs))

#################################################### iterate if you need
torch.manual_seed(777)
hvs_qehvi= []

# call helper functions to generate initial training data and initialize model
train_x_qehvi, train_obj_qehvi = generate_initial_data()
print(train_x_qehvi)
mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

# compute pareto front
pareto_mask = is_non_dominated(train_obj_qehvi)
pareto_y = train_obj_qehvi[pareto_mask]

# compute hypervolume
volume = hv.compute(pareto_y)
hvs_qehvi.append(volume)


################# run N_BATCH rounds of BayesOpt after the initial random batch
t0 = time.time()

# fit the models
fit_gpytorch_model(mll_qehvi)

# define the qEI and qNEI acquisition modules using a QMC sampler
qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

# optimize acquisition functions and get new observations
new_x_qehvi= optimize_qehvi_and_get_observation(
    model_qehvi, train_obj_qehvi, qehvi_sampler
)


## optimize acquisition functions and get new observations
#new_x_qehvi, new_obj_qehvi = optimize_qehvi_and_get_observation(
#    model_qehvi, train_obj_qehvi, qehvi_sampler
#)

## update training points
#train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
#train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])

## update progress
## compute hypervolume
#pareto_mask = is_non_dominated(train_obj_qehvi)
#pareto_y = train_obj_qehvi[pareto_mask]

## compute hypervolume
#volume = hv.compute(pareto_y)
#hvs_qehvi.append(volume)

## reinitialize the models so they are ready for fitting on next iteration
## Note: we find improved performance from not warm starting the model hyperparameters
## using the hyperparameters from the previous iteration
#mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

#t1 = time.time()
################# run N_BATCH rounds of BayesOpt after the initial random batch

#t1 = time.time()
#train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
#train_obj_qehvi = torch.cat([train_obj_qehvi, new_x_qehvi])
#x_all=pd.DataFrame(train_x_qehvi.cpu().numpy())
#y_all=pd.DataFrame(train_obj_qehvi.cpu().numpy())

#result = pd.concat([x_all,y_all], axis=1)
#result.columns=['x1','x2','x3','y1','y2']
#result.to_csv('data.csv',index=False)
#print(new_x_qehvi)
