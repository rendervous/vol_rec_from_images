import torch
import os
import rendervous as rdv
import matplotlib.pyplot as plt
from _tools import psnr


DATASET = 'cloud'

# Reconstructions to compare
reconstructions = [
    f'./reconstructions/{DATASET}_drt_ms_DRT.pt',
    f'./reconstructions/{DATASET}_sps_ms_SPS.pt'
]

device = rdv.device()

# Load environment image
environment_tensor = torch.load(f"./data/environment00.pt", map_location=device).contiguous()
# Load reference images from dataset
dataset = torch.load(f'./datasets/{DATASET}_dataset.pt', map_location=device)
# Load reference volume
reference_volume = torch.load(f'./data/{DATASET}_grid.pt', map_location=device)
reference_volume *= dataset['sigma_scale']  # Real volume used is scaled.

reference_images = dataset['radiances']
camera_poses = dataset['cameras']


TEST_IMAGE = 69  # 64-79 are the generated testing images

# Tool method to render a volume with dataset configurations
def render_volume(sigma_tensor, samples=16*1024):
    ds = rdv.DependencySet()
    sigma_scale = dataset['sigma_scale']
    scattering_albedo = dataset['scattering_albedo']
    phase_g = dataset['phase_g']
    ds.add_parameters(
        sigma_tensor=sigma_tensor,
        scattering_albedo_tensor=torch.tensor([scattering_albedo, scattering_albedo, scattering_albedo], device=device),
        phase_g_tensor=torch.tensor([phase_g], device=device),
        environment_tensor=environment_tensor,
        camera_poses=camera_poses[TEST_IMAGE: TEST_IMAGE+1]
    )
    ds.requires(rdv.medium_transmittance_RT)
    ds.requires(rdv.medium_radiance_path_integrator_NEE_DT)
    ds.requires(rdv.camera_sensors, width=512, height=512, jittered=True)

    samples_per_group = min(samples, 8)
    samples = ((samples + samples_per_group - 1) // samples_per_group) * samples_per_group
    radiances = torch.zeros(1, 512, 512, 3, device=device)
    groups = samples // samples_per_group
    print('[INFO] Rendering')
    with torch.no_grad():
        for i in range(groups):
            im = ds.camera.capture(ds.radiance, fw_samples=samples_per_group, batch_size=512 * 512)
            torch.add(radiances, im, alpha=1.0 / groups, out=radiances)
    return radiances

import matplotlib
norm = matplotlib.colors.PowerNorm(1.0/4, vmin=0, vmax=dataset['sigma_scale'], clip=True)

# Tool method to draw a volume slice
def draw_volume(sigma_tensor, axis):
    axis.imshow(sigma_tensor[:, :, sigma_tensor.shape[2]//2, 0].T.cpu().numpy(), cmap='bone', norm=norm)
    axis.invert_yaxis()
    axis.axis('off')

def draw_volume_difference(sigma_tensor, axis):
    volume_diff = (reference_volume-sigma_tensor).abs()
    axis.imshow(volume_diff[:, :, sigma_tensor.shape[2]//2, 0].T.cpu().numpy(), vmin=0, vmax=dataset['sigma_scale'], cmap='hot')
    axis.invert_yaxis()
    axis.axis('off')

fig, axes = plt.subplots(2, 1 + len(reconstructions), figsize=((1+len(reconstructions))*4.05, 4.2), dpi=300)

draw_volume(reference_volume, axes[0,0])
axes[1,0].axis('off')

for i in range(1, len(reconstructions)+1):
    rec = torch.load(reconstructions[i-1], map_location=device)
    sigma_tensor = rec['sigma_grid']  # unnormalized
    draw_volume(sigma_tensor, axes[0, i])
    axes[0, i].text(3, 3, f"PSNR: {psnr(0.8 * reference_volume / dataset['sigma_scale'], 0.8 * sigma_tensor / dataset['sigma_scale']):0.2f}", color='white')
    draw_volume_difference(sigma_tensor, axes[1, i])
    axes[1, i].text(3, 3, os.path.basename(reconstructions[i-1]), color='white')

fig.tight_layout(pad=0.1)
plt.show()


fig, axes = plt.subplots(2, 1 + len(reconstructions), figsize=((1+len(reconstructions))*4.05, 2*4.05), dpi=300)

axes[0, 0].imshow(reference_images[TEST_IMAGE].cpu() ** (1.0/2.2))
axes[0, 0].invert_yaxis()
axes[0, 0].axis('off')
axes[1, 0].axis('off')

for i in range(1, len(reconstructions)+1):
    rec = torch.load(reconstructions[i-1], map_location=device)
    sigma_tensor = rec['sigma_grid']  # unnormalized
    rec_image = render_volume(sigma_tensor)
    axes[0, i].imshow(rec_image[0].cpu()  ** (1.0/2.2))
    axes[0, i].invert_yaxis()
    axes[0, i].axis('off')
    axes[0, i].text(8, 8, f"PSNR: {psnr(reference_images[TEST_IMAGE], rec_image[0]):0.2f}", color='white')

    axes[1, i].imshow((rec_image[0] - reference_images[TEST_IMAGE]).abs().mean(dim=-1).cpu(), vmin=0, vmax=1.0, cmap='hot')
    axes[1, i].invert_yaxis()
    axes[1, i].axis('off')
    axes[1, i].text(3, 3, os.path.basename(reconstructions[i-1]), color='white')



fig.tight_layout(pad=0.05)
plt.show()
