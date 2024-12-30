#%%

import os
import torch
from torch.distributions import LowRankMultivariateNormal

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
MOBO_TEST = os.environ.get("MOBO_TEST")

#%% Test function change

from botorch.test_functions.multi_objective import DTLZ1
d = 2
M = 2
problem = DTLZ1(dim=d, num_objectives=M, negate=True).to(**tkwargs)

#%%


from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples


def generate_initial_data(n=6):
    # generate training data
    train_x = draw_sobol_samples(
        bounds=problem.bounds,n=1, q=n, seed=torch.randint(1000000, (1,)).item()
    ).squeeze(0)
    train_obj = problem(train_x)
    return train_x, train_obj


def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

#%%

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex

standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds[1] = 1

BATCH_SIZE = 2
NUM_RESTARTS = 20
RAW_SAMPLES = 1024

def optimize_qehvi_and_get_observation(model, train_obj, sampler):
    partitioning = NondominatedPartitioning(num_outcomes=problem.num_objectives, Y=train_obj)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(), # use known reference point
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
        options={"batch_limit":5, "maxiter":200, "nonnegative":True},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_x)
    return new_x, new_obj

#%%

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
MC_SAMPLES = 128
verbose = False

hvs_qehvi_all = []

hv = Hypervolume(ref_point=problem.ref_point)

# average over multiple trials
for trial in range(1, N_TRIALS + 1):
    torch.manual_seed(trial)

    hvs_qehvi= []

    # call helper functions to generate initial training data and initialize model
    train_x_qehvi, train_obj_qehvi = generate_initial_data(n=6)
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

    # compute pareto front
    pareto_mask = is_non_dominated(train_obj_qehvi)
    pareto_y = train_obj_qehvi[pareto_mask]
    # compute hypervolume

    volume = hv.compute(pareto_y)

    hvs_qehvi.append(volume)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):

        t0 = time.time()

        # fit the models
        fit_gpytorch_model(mll_qehvi)

        # define the qEI and qNEI acquisition modules using a QMC sampler
        qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

        # optimize acquisition functions and get new observations
        new_x_qehvi, new_obj_qehvi = optimize_qehvi_and_get_observation(
            model_qehvi, train_obj_qehvi, qehvi_sampler
        )

        # update training points
        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])

        # update progress
        # compute hypervolume
        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]
        # compute hypervolume
        volume = hv.compute(pareto_y)
        hvs_qehvi.append(volume)

        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

        t1 = time.time()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: Hypervolume (qEHVI) = "
                f"({hvs_qehvi[-1]:>4.2f}), "
                f"time = {t1-t0:>4.2f}.", end=""
            )
        else:
            print(".", end="")

    hvs_qehvi_all.append(hvs_qehvi)

#%%

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

fig, axes = plt.subplots(1, 1, figsize=(8, 6))
algos = ["qEHVI"]
cm = plt.cm.get_cmap('viridis')

batch_number = torch.cat(
    [torch.zeros(6), torch.arange(1, N_BATCH+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
).numpy()

axes.set_xlabel("Objective 1")
axes.set_ylabel("Objective 2")

sc = axes.scatter(-1*train_obj_qehvi[:,  0].cpu().numpy(), -1*train_obj_qehvi[:, 1].cpu().numpy(), c=batch_number, alpha=0.8)

norm = plt.Normalize(batch_number.min(), batch_number.max())
sm = ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")

plt.savefig('test_DTLZ1.png', dpi=300)

#%%

