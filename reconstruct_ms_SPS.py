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

plt.imshow(original_tensor[:,:,original_tensor.shape[2]//2,0].T.detach().cpu().numpy(), vmin=0.0, vmax=1.25)
plt.gca().invert_yaxis()
plt.show()

START_LR = 0.008
LR_DECAY = 0.8
ITERATIONS = 400
LEVELS = 3
BW_TECHNIQUE = 'SPS'  # DRT, SPS, DRTDS, DRTQ, DT, DS

DOWNSCALES = [1 << (LEVELS - l - 1) for l in range(LEVELS)]
ITERATION_PER_STAGES = [ITERATIONS // (1 << (LEVELS - l)) for l in range(LEVELS)]
ITERATION_PER_STAGES[0] += ITERATIONS - sum(ITERATION_PER_STAGES)
# Enhancing alpha used with EMA. Low resolution reconstruction requires larger values to avoid bias
ENHANCING = 0.01 if original_tensor.shape[0] * original_tensor.shape[1] * original_tensor.shape[
    2] > 128 * 128 * 128 else 0.1

training_radiances = dataset['radiances'][:64]
training_camera_poses = dataset['cameras'][:64]


# Load initial state from emission-absorption model
ea_reconstruction = torch.load(f'./reconstructions/{DATASET}_ae_3.pt', map_location=device)
initial_sigmas = ea_reconstruction['sigma_grid']
initial_sigmas[initial_sigmas < 0.1251] = 0.0
plt.imshow(initial_sigmas[:, :, initial_sigmas.shape[2]//2, 0].T.detach().cpu(), vmin=0, vmax=sigma_scale)
plt.gca().invert_yaxis()
plt.show()

# create mask from initial estimate
final_mask = rdv.resample_grid((initial_sigmas > 0.1251).float(), tuple((d - 1) // 2 + 1 for d in original_tensor.shape[:-1]))
plt.imshow(final_mask[:, :, final_mask.shape[2]//2, 0].T.detach().cpu(), vmin=0, vmax=1.0)
plt.gca().invert_yaxis()
plt.show()

current_stage_lr = START_LR
previous_solution_sigma = initial_sigmas
global_step = 0
start_time = time.perf_counter()
history = None  # history is initialized within the first stage with the first initial state.

for level, (downscale, iterations) in enumerate(zip(DOWNSCALES, ITERATION_PER_STAGES)):
    previous_solution_sigma, history = _tools.SPS_multiscattering_reconstruction(
        radiances=training_radiances,
        camera_poses=training_camera_poses,
        environment=environment_tensor,
        scattering_albedo=dataset['scattering_albedo'],
        phase_g=dataset['phase_g'],
        optimal_shape=original_tensor.shape[:-1],
        sigma_resolution=tuple((d - 1)//downscale for d in original_tensor.shape[:-1]),
        sigma_scale=sigma_scale,
        initial_sigma_value=previous_solution_sigma,
        sigma_mask=final_mask,
        bw_mode=BW_TECHNIQUE,
        lr=current_stage_lr,
        lr_decay=LR_DECAY,
        fw_samples=16,
        bw_samples=6 if BW_TECHNIQUE == 'SPS' else 4,
        iterations=iterations,
        global_step=global_step,
        history=history,
        enhancing_alpha=ENHANCING ** ((level + 1)/LEVELS)
    )
    current_stage_lr *= LR_DECAY
    global_step += iterations

    plt.imshow(previous_solution_sigma[:,:,previous_solution_sigma.shape[2]//2,0].T.detach().cpu().numpy(), vmin=0.0, vmax=sigma_scale)
    plt.gca().invert_yaxis()
    plt.show()

print(f'[INFO] Finished in {time.perf_counter() - start_time}s')
print(f'[INFO] Final PSNR: {_tools.psnr(original_tensor * 0.8, previous_solution_sigma / sigma_scale)}')

if not os.path.exists('./reconstructions'):
    os.mkdir('./reconstructions')

torch.save({
    'dataset': DATASET,
    'sigma_grid': previous_solution_sigma.detach().cpu(),
}, f'./reconstructions/{DATASET}_sps_ms_{BW_TECHNIQUE}.pt')