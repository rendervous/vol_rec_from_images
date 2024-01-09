import torch
import rendervous as rdv
import _tools
import os


# settings
DATASET = 'disney'
BAN_EMITTERS = False
BLURRED = True
SAMPLES = 128
CAMERAS = 240
CAMERA_WIDTH = 512
CAMERA_HEIGHT = 512
SIGMA_SCALE = 100
PHASE_G = 0.4
SCATTERING_ALBEDO = 0.99


# rdv device
device = rdv.device()

# load tensors
sigma_tensor = torch.load(f'./data/{DATASET}_grid.pt', map_location=device)
environment_tensor = torch.load(f"./data/{'blur_' if BLURRED else ''}environment00.pt", map_location=device).contiguous()

camera_poses = torch.load(f"./datasets/{'blur_' if BLURRED else ''}{DATASET}_dataset{'' if BAN_EMITTERS else '_full'}.pt", map_location=device)['cameras'] # _tools.generate_camera_poses(CAMERAS, (environment_tensor > 3.14159 * 2).float())

ds = rdv.DependencySet()
ds.add_parameters(
    sigma_tensor=lambda: sigma_tensor * SIGMA_SCALE,
    scattering_albedo_tensor=torch.tensor([SCATTERING_ALBEDO, SCATTERING_ALBEDO, SCATTERING_ALBEDO], device=device),
    phase_g_tensor=torch.tensor([PHASE_G], device=device),
    environment_tensor=environment_tensor,
    camera_poses=camera_poses
)
ds.requires(rdv.medium_transmittance_DDA)
ds.requires(rdv.medium_radiance_transmitted)
ds.requires(rdv.camera_sensors, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, jittered=True)

samples_per_group = min(SAMPLES, 8)
samples = ((SAMPLES + samples_per_group - 1) // samples_per_group) * samples_per_group
radiances = torch.zeros(CAMERAS, CAMERA_HEIGHT, CAMERA_WIDTH, 3, device=device)
groups = samples // samples_per_group
print('[INFO] Rendering')
import time
from datetime import timedelta
with torch.no_grad():
    t = time.perf_counter()
    for i in range(groups):
        im = ds.camera.capture(ds.radiance, fw_samples=samples_per_group, batch_size=CAMERA_HEIGHT*CAMERA_WIDTH)
        torch.add(radiances, im, alpha=1.0 / groups, out=radiances)
        elapsed = time.perf_counter() - t
        if i == 0 or i == groups-1:  # render pos for the first and final only
            import matplotlib.pyplot as plt
            plt.imshow(im[0].detach().cpu() ** (1.0/2.2))
            plt.show()
        print(f'[INFO] Completed {100 * (i + 1) / groups}% ETA: {timedelta(seconds=elapsed  * (groups - i - 1)/ (i+1))}')

if not os.path.exists('./datasets'):
    os.mkdir('./datasets')

torch.save({
    'radiances': radiances.detach().cpu(),
    'cameras': camera_poses.detach().cpu(),
    'samples': SAMPLES,
    'sigma_scale': SIGMA_SCALE,
    'phase_g': PHASE_G,
    'scattering_albedo': SCATTERING_ALBEDO
}, f"./datasets/{'blur_' if BLURRED else ''}{DATASET}_dataset{'' if BAN_EMITTERS else '_full'}_ao.pt")
