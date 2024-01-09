import rendervous as rdv
import torch
import _tools
import matplotlib.pyplot as plt
import os
import time


DATASET = 'cloud'

# Load dataset
device = rdv.device()
dataset = torch.load(f'./datasets/{DATASET}_dataset.pt', map_location=device)
original_tensor = torch.load(f'./data/{DATASET}_grid.pt', map_location=device)
sigma_scale = dataset['sigma_scale'] * 1.25  # this is not mandatory but is helpful to normalize the densities and have a faster convergence
environment_tensor = torch.load('./data/environment00.pt')

START_LR = 0.01
LR_DECAY = 0.8
ITERATIONS = 2000
LEVELS = 4
SH_LEVELS = 3

DOWNSCALES = [1 << (LEVELS - l - 1) for l in range(LEVELS)]
ITERATION_PER_STAGES = [ITERATIONS // (1 << (LEVELS - l)) for l in range(LEVELS)]
ITERATION_PER_STAGES[0] += ITERATIONS - sum(ITERATION_PER_STAGES)

training_radiances = dataset['radiances'][:64]
training_camera_poses = dataset['cameras'][:64]

current_stage_lr = START_LR
previous_solution_sigma = None
previous_solution_emission = None
global_step = 0
start_time = time.perf_counter()

for downscale, iterations in zip(DOWNSCALES, ITERATION_PER_STAGES):
    previous_solution_sigma, previous_solution_emission = _tools.absorption_emission_reconstruction(
        radiances=training_radiances,
        camera_poses=training_camera_poses,
        environment=environment_tensor,
        optimal_shape=original_tensor.shape[:-1],
        sigma_resolution=tuple((d - 1)//downscale for d in original_tensor.shape[:-1]),
        sigma_scale=sigma_scale,
        emission_resolution=tuple((d - 1) // downscale // 2 for d in original_tensor.shape[:-1]),
        emission_sh_levels=SH_LEVELS,
        initial_sigma_value=1.0 if previous_solution_sigma is None else previous_solution_sigma,
        initial_emission_value=1.0 if previous_solution_emission is None else previous_solution_emission,
        lr=current_stage_lr,
        lr_decay=LR_DECAY,
        iterations=iterations,
        global_step=global_step
    )
    current_stage_lr *= LR_DECAY
    global_step += iterations

    plt.imshow(previous_solution_sigma[:,:,previous_solution_sigma.shape[2]//2,0].T.detach().cpu().numpy(), vmin=0.0, vmax=sigma_scale)
    plt.gca().invert_yaxis()
    plt.show()

    plt.imshow( previous_solution_emission[:,:, previous_solution_emission.shape[2]//2, [0,1,2]].permute(1,0,2).detach().cpu().numpy())
    plt.gca().invert_yaxis()
    plt.show()

print(f'[INFO] Finished in {time.perf_counter() - start_time}s')
print(f'[INFO] Final PSNR: {_tools.psnr(original_tensor * 0.8, previous_solution_sigma / sigma_scale)}')

if not os.path.exists('./reconstructions'):
    os.mkdir('./reconstructions')

torch.save({
    'dataset': DATASET,
    'sigma_grid': previous_solution_sigma.detach().cpu(),
    'emission_tensor': previous_solution_emission.detach().cpu(),
}, f'./reconstructions/{DATASET}_ae_{SH_LEVELS}.pt')