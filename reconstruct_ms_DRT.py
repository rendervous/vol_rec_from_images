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
# We reconstruct assuming the max value to be upto 1.25 of the original.
sigma_scale = dataset['sigma_scale'] * 1.25  # this is not mandatory but is helpful to normalize the densities and have a faster convergence
environment_tensor = torch.load('./data/environment00.pt')

START_LR = 0.0025
LR_DECAY = 0.6
ITERATIONS = 10000
LEVELS = 4
BW_TECHNIQUE = 'DRT'  # DRT, SPS, DRTDS, DRTQ, DT, DS

DOWNSCALES = [1 << (LEVELS - l - 1) for l in range(LEVELS)]
ITERATION_PER_STAGES = [ITERATIONS // (1 << (LEVELS - l)) for l in range(LEVELS)]
ITERATION_PER_STAGES[0] += ITERATIONS - sum(ITERATION_PER_STAGES)

training_radiances = dataset['radiances'][:64]
training_camera_poses = dataset['cameras'][:64]

current_stage_lr = START_LR
previous_solution_sigma = None
global_step = 0
start_time = time.perf_counter()

for downscale, iterations in zip(DOWNSCALES, ITERATION_PER_STAGES):
    previous_solution_sigma = _tools.DRT_multiscattering_reconstruction(
        radiances=training_radiances,
        camera_poses=training_camera_poses,
        environment=environment_tensor,
        scattering_albedo=dataset['scattering_albedo'],
        phase_g=dataset['phase_g'],
        optimal_shape=original_tensor.shape[:-1],
        sigma_resolution=tuple((d - 1)//downscale for d in original_tensor.shape[:-1]),
        sigma_scale=sigma_scale,
        initial_sigma_value=0.1 if previous_solution_sigma is None else previous_solution_sigma,
        bw_mode=BW_TECHNIQUE,
        lr=current_stage_lr,
        lr_decay=LR_DECAY,
        fw_samples=256,
        bw_samples=20 if BW_TECHNIQUE == 'SPS' else 16,
        iterations=iterations,
        global_step=global_step
    )
    current_stage_lr *= LR_DECAY
    global_step += iterations

    plt.imshow(previous_solution_sigma[:,:,previous_solution_sigma.shape[2]//2,0].T.detach().cpu().numpy(), vmin=0.0, vmax=sigma_scale)
    plt.gca().invert_yaxis()
    plt.show()

print(f'[INFO] Finished in {time.perf_counter() - start_time}s')
# Here is 0.8 = 1/1.25 to scale the density field back to the original range 0..1 wrt to the reconstruction.
print(f'[INFO] Final PSNR: {_tools.psnr(original_tensor * 0.8, previous_solution_sigma / sigma_scale)}')

if not os.path.exists('./reconstructions'):
    os.mkdir('./reconstructions')

torch.save({
    'dataset': DATASET,
    'sigma_grid': previous_solution_sigma.detach().cpu(),
}, f'./reconstructions/{DATASET}_drt_ms_{BW_TECHNIQUE}.pt')