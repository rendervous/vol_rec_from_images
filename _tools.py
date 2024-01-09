import torch
import numpy as np
import rendervous as rdv
from typing import Tuple, Union, List, Callable, Optional
import time


# Grids or image metrics

def psnr(grid1, grid2):
    if isinstance(grid1, torch.Tensor):
        return -10 * torch.log10(((grid1 - grid2)**2).mean()).item()
    return -10 * np.log10(((grid1 - grid2)**2).mean())


def rmse(grid1, grid2):
    return np.sqrt(((grid1 - grid2) ** 2).mean().item())


def mae(grid1, grid2):
    return ((grid1 - grid2).abs()).mean().item()


def nmae(grid1, grid2):
    return mae(grid1, grid2) / (grid2.abs().mean() + 0.00000001).item()


def nrmse(grid1, grid2):
    return rmse(grid1, grid2) / (grid2.abs().mean() + 0.00000001).item()


def get_all_measurements(current, target):
    return dict(
        psnr = psnr(current, target),
        rmse = rmse(current, target),
        mae = mae(current, target),
        nmae = nmae(current, target),
        nrmse = nrmse(current, target),
    )


# Generate camera poses without direct capture of environment emitters

def generate_camera_poses(number_of_cameras: int, environment_mask: torch.Tensor) -> torch.Tensor:
    def _is_banned_environment_view(camera_pose: torch.Tensor):
        try_ds = rdv.DependencySet()
        try_ds.requires(rdv.medium_box_AABB)
        try_ds.add_parameters(sigma=rdv.constant(3, 0.0))
        try_ds.add_parameters(majorant=rdv.constant(6, 0.000001, 1000.0))
        try_ds.add_parameters(environment_tensor=environment_mask)
        try_ds.requires(rdv.medium_radiance_transmitted)
        try_ds.add_parameters(camera_poses=camera_pose)
        try_ds.requires(rdv.camera_sensors, width=128, height=128)
        camera = try_ds.camera
        radiance = try_ds.radiance
        im = camera.capture(radiance)
        return im.sum() > 0.0  # not all is black
    device = rdv.device()
    try_camera_poses_tensor = rdv.random_equidistant_camera_poses(2 * number_of_cameras, radius=2.0)
    camera_poses = torch.zeros(number_of_cameras, 9, device=device)
    print('[INFO] Filtering camera poses')
    c = 0
    for i in range(2 * number_of_cameras):
        # print(f'[INFO] trying {try_camera_poses_tensor[i]}')
        if not _is_banned_environment_view(try_camera_poses_tensor[i:i+1]):
            camera_poses[c] = try_camera_poses_tensor[i]
            c += 1
        if c >= number_of_cameras:
            break
    if c < number_of_cameras:
        raise Exception('Poses could not avoid higher emittance')
    return camera_poses


# Create optimization objects with common setup

def create_optimization_objects(
        parameters: torch.Tensor,
        optimizer_id: str,
        scheduler_id: str,
        lr: float,
        lr_decay: float,
        iterations: int
):
    optimizer_args = dict()
    if optimizer_id == 'adam':
        optimizer_type = torch.optim.Adam
        optimizer_args = dict(eps=1e-15)
    elif optimizer_id == 'adam_reg':
            optimizer_type = torch.optim.Adam
            optimizer_args = dict(weight_decay=0.0000001)
    else:
        raise NotImplemented()
    optimizer = optimizer_type([parameters], lr=lr, **optimizer_args)
    if scheduler_id == 'none':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [0], 1.0)
    elif scheduler_id == 'steps':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            [iterations * (i + 1) // 10 for i in range(9)],
            gamma=lr_decay ** 0.1)
    else:
        raise NotImplemented()

    return optimizer, scheduler


def reconstruction_step(
        ds: rdv.DependencySet,
        radiances: torch.Tensor,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimization_objects: List[Tuple],
        regularizers: List,
        *,
        fw_samples: int = 1,
        bw_samples: int = 1,
        sensors_batch: Optional[torch.Tensor] = None,
        no_update: bool = False,
        history: torch.Tensor = None,
        enhancing: float = 0.1,
):
    if history is not None:
        assert sensors_batch is None, 'Not supported batches if EMA is applied'
        assert history.shape == radiances.shape, 'History radiances dimension should match radiance dimension'
    for o, _ in optimization_objects:
        o.zero_grad()
    ds.forward_dependency()
    camera = ds.camera
    radiance = ds.radiance
    if sensors_batch is not None:
        batch_size = len(sensors_batch)
        batch_radiances = radiances[sensors_batch[:,0], sensors_batch[:,1], sensors_batch[:,2]]
    else:
        batch_size = camera.index_shape[1] * camera.index_shape[2]
        batch_radiances = radiances
    rendered = camera.capture(
        radiance,
        sensors_batch=sensors_batch,
        batch_size=batch_size,
        fw_samples=fw_samples,
        bw_samples=bw_samples
    )

    if history is not None:
        # EMA appliance. enhance_output bypass the enhancement process in the backward call
        rendered = rdv.enhance_output(rendered, enhance_process=lambda o: torch.add(history * (1 - enhancing), alpha=enhancing, other=o, out=history))

    loss = loss_function(rendered, batch_radiances)
    loss_value = loss.item()
    for r in regularizers:
        loss = loss + r()
    loss.backward()
    ds.backward_dependency()
    if not no_update:
        for o, s in optimization_objects:
            o.step()
            s.step()
    return loss_value


def reconstruction_stage(
        ds: rdv.DependencySet,
        radiances: torch.Tensor,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimization_objects: List[Tuple],
        regularizers: List,
        after_step: Callable[..., None],
        *,
        fw_samples: int = 1,
        bw_samples: int = 1,
        steps: int = 6000,
        batch_size: Optional[int] = None,
        history: torch.Tensor = None,
        enhancing: float = 0.1,
):
    if history is not None:
        assert batch_size is None, 'Batches are not supported when EMA is applied'
    sensors_batch = None
    for step in range(steps):
        t = time.perf_counter()
        if batch_size is not None:
            with torch.no_grad():
                sensors_batch = ds.camera.random_sensors(batch_size=batch_size, out=sensors_batch)
        loss_value = reconstruction_step(
            ds,
            radiances,
            loss_function,
            optimization_objects,
            regularizers,
            fw_samples=fw_samples,
            bw_samples=bw_samples,
            sensors_batch=sensors_batch,
            history=history,
            enhancing=enhancing
        )
        after_step(
            iteration=step,
            loss=loss_value,
            duration=time.perf_counter() - t,
        )


def absorption_emission_reconstruction(
        radiances: torch.Tensor,
        camera_poses: torch.Tensor,
        environment: torch.Tensor,
        optimal_shape: List[int],
        sigma_resolution: Tuple[int, ...],
        sigma_scale: float,  # good for normalization
        emission_resolution: Tuple[int, ...],
        emission_sh_levels: int,
        initial_sigma_value: Union[float, torch.Tensor],
        initial_emission_value: Union[float, torch.Tensor],
        lr: float,
        lr_decay: float,
        iterations: int,
        global_step: int,
):
    # Create grid parameters
    sigma_parameters = rdv.tensor(*(d + 1 for d in sigma_resolution), 1).zero_()
    emission_coefficients = emission_sh_levels ** 2
    emission_parameters = rdv.tensor(*(d + 1 for d in emission_resolution), emission_coefficients*3).zero_()
    # initialize
    with torch.no_grad():
        if isinstance(initial_sigma_value, float):
            sigma_parameters.fill_(initial_sigma_value / sigma_scale)
        else:
            sigma_parameters = rdv.resample_grid(initial_sigma_value / sigma_scale, sigma_parameters.shape[:-1])
        if isinstance(initial_emission_value, float):
            for c in range(3):
                emission_parameters[..., c * emission_coefficients] = initial_emission_value
        else:
            emission_parameters = rdv.resample_grid(initial_emission_value, emission_parameters.shape[:-1])
    sigma_parameters = torch.nn.Parameter(sigma_parameters)
    emission_parameters = torch.nn.Parameter(emission_parameters)

    # Create dependency set to create radiance map and camera sensor objects
    ds = rdv.DependencySet()
    ds.requires(rdv.medium_box_normalized, t=optimal_shape)
    ds.add_parameters(
        sigma_tensor=lambda: rdv.dclamp(sigma_parameters, 0.001, 1.0) * sigma_scale,
        emission_tensor=emission_parameters,
        environment_tensor=environment,
        camera_poses=camera_poses
    )
    ds.requires(rdv.medium_sigma_tensor)
    ds.requires(rdv.medium_emission_tensor)
    ds.requires(rdv.medium_exitance_radiance_emission)
    ds.requires(rdv.medium_radiance_collision_integrator_DDA)
    ds.requires(rdv.camera_sensors, width=radiances.shape[-2], height=radiances.shape[-3], jittered=True)

    optimization_objects = [
        create_optimization_objects(sigma_parameters, 'adam', 'steps', lr, lr_decay, iterations),
        create_optimization_objects(emission_parameters, 'adam', 'steps', lr * 30 / emission_coefficients, lr_decay, iterations)
    ]

    regularizers = [
        lambda: rdv.total_variation(sigma_parameters).mean() * 0.00025,    # smooth prior for sigma
        lambda: rdv.total_variation(emission_parameters).mean() * 0.00005,   # smooth prior for emission field
        lambda: (emission_parameters ** 2).mean() * 0.0001,                  # avoid emission overfit (L2 reg)
    ]

    def after_step(**stats):
        with torch.no_grad():
            # apply constraint if required!
            # sigma_parameter.clamp_min_(0.001)
            stats = {**stats}
            it = stats['iteration']
            if (global_step + it + 1) % max(1, iterations // 50) == 0:
                print(str(global_step + it + 1) + "- " + (
                    '\t'.join(f"{k}:{'{a:.6f}'.format(a=v)} " for k, v in stats.items() if isinstance(v, float))))

    reconstruction_stage(
        ds=ds,
        radiances=radiances,
        loss_function=torch.nn.HuberLoss(),
        optimization_objects=optimization_objects,
        regularizers=regularizers,
        after_step=after_step,
        fw_samples=4,
        bw_samples=1,
        steps=iterations,
        batch_size=64*1024
    )

    ds.forward_dependency()
    return ds.sigma_tensor.detach().clone(), ds.emission_tensor.detach().clone()


def DRT_multiscattering_reconstruction(
        radiances: torch.Tensor,
        camera_poses: torch.Tensor,
        environment: torch.Tensor,
        scattering_albedo: float,
        phase_g: float,
        optimal_shape: List[int],
        sigma_resolution: Tuple[int, ...],
        sigma_scale: float,  # good for normalization
        initial_sigma_value: Union[float, torch.Tensor],
        bw_mode: str,
        lr: float,
        lr_decay: float,
        iterations: int,
        fw_samples: int,
        bw_samples: int,
        global_step: int,
):
    # Create grid parameters
    sigma_parameters = rdv.tensor(*(d + 1 for d in sigma_resolution), 1).zero_()
    # initialize
    with torch.no_grad():
        if isinstance(initial_sigma_value, float):
            sigma_parameters.fill_(initial_sigma_value / sigma_scale)
        else:
            sigma_parameters = rdv.resample_grid(initial_sigma_value / sigma_scale, sigma_parameters.shape[:-1])
    sigma_parameters = torch.nn.Parameter(sigma_parameters)

    # Create dependency set to create radiance map and camera sensor objects
    ds = rdv.DependencySet()
    ds.requires(rdv.medium_box_normalized, t=optimal_shape)
    ds.add_parameters(
        sigma_tensor=lambda: sigma_parameters * sigma_scale,
        scattering_albedo=rdv.constant(3, scattering_albedo, scattering_albedo, scattering_albedo), # always white in the experiments
        emission=rdv.constant(6, 0.0, 0.0, 0.0),
        phase_g=rdv.constant(3, phase_g),
        environment_tensor=environment,
        camera_poses=camera_poses
    )
    ds.requires(rdv.medium_sigma_tensor)
    ds.requires(rdv.medium_transmittance_RT)
    # build the MO integrator
    if bw_mode == 'DT':
        ds.requires(rdv.medium_radiance_path_integrator_DT)
    elif bw_mode == 'DS':
        ds.requires(rdv.medium_radiance_path_integrator_NEE_DT)
    elif bw_mode == 'DRTDS':
        ds.requires(rdv.medium_radiance_path_integrator_NEE_DRTDS)
    elif bw_mode == 'DRT':
        ds.requires(rdv.medium_radiance_path_integrator_NEE_DRT)
    elif bw_mode == 'DRTQ':
        ds.requires(rdv.medium_radiance_path_integrator_NEE_DRTQ)
    elif bw_mode == 'SPS':
        ds.requires(rdv.medium_radiance_path_integrator_NEE_SPS)
    else:
        raise NotImplemented()
    ds.requires(rdv.camera_sensors, width=radiances.shape[-2], height=radiances.shape[-3], jittered=True)

    optimization_objects = [
        create_optimization_objects(sigma_parameters, 'adam', 'steps', lr, lr_decay, iterations),
    ]

    regularizers = [  # There werent used regularizers in DRT
    ]

    def after_step(**stats):
        with torch.no_grad():
            # apply constraint if required!
            sigma_parameters.clamp_(0.0, 1.0)

            stats = {**stats}
            it = stats['iteration']
            if (global_step + it + 1) % max(1, iterations // 50) == 0:
                print(str(global_step + it + 1) + "- " + (
                    '\t'.join(f"{k}:{'{a:.6f}'.format(a=v)} " for k, v in stats.items() if isinstance(v, float))))

    reconstruction_stage(
        ds=ds,
        radiances=radiances,
        loss_function=torch.nn.L1Loss(),
        optimization_objects=optimization_objects,
        regularizers=regularizers,
        after_step=after_step,
        fw_samples=fw_samples,
        bw_samples=bw_samples,
        steps=iterations,
        batch_size=32*1024
    )

    ds.forward_dependency()
    return ds.sigma_tensor.detach().clone()


def SPS_multiscattering_reconstruction(
        radiances: torch.Tensor,
        camera_poses: torch.Tensor,
        environment: torch.Tensor,
        scattering_albedo: float,
        phase_g: float,
        optimal_shape: List[int],
        sigma_resolution: Tuple[int, ...],
        sigma_scale: float,  # good for normalization
        initial_sigma_value: Union[float, torch.Tensor],
        sigma_mask: torch.Tensor,
        bw_mode: str,
        lr: float,
        lr_decay: float,
        iterations: int,
        fw_samples: int,
        bw_samples: int,
        global_step: int,
        history: Optional[torch.Tensor],
        enhancing_alpha: float
):
    # Create grid parameters
    sigma_parameters = rdv.tensor(*(d + 1 for d in sigma_resolution), 1).zero_()
    # initialize
    with torch.no_grad():
        if isinstance(initial_sigma_value, float):
            sigma_parameters.fill_(initial_sigma_value / sigma_scale)
        else:
            sigma_parameters = rdv.resample_grid(initial_sigma_value / sigma_scale, sigma_parameters.shape[:-1])
    sigma_parameters = torch.nn.Parameter(sigma_parameters)

    # Create rescaled mask to this stage
    rescaled_mask = rdv.resample_grid(sigma_mask, sigma_parameters.shape[:-1]) * sigma_scale

    # Create dependency set to create radiance map and camera sensor objects
    ds = rdv.DependencySet()
    ds.requires(rdv.medium_box_normalized, t=optimal_shape)
    ds.add_parameters(
        sigma_tensor=lambda: rdv.dclamp(sigma_parameters) * rescaled_mask,
        scattering_albedo=rdv.constant(3, scattering_albedo, scattering_albedo, scattering_albedo), # always white in the experiments
        emission=rdv.constant(6, 0.0, 0.0, 0.0),
        phase_g=rdv.constant(3, phase_g),
        environment_tensor=environment,
        camera_poses=camera_poses
    )
    ds.requires(rdv.medium_sigma_tensor)
    ds.requires(rdv.medium_transmittance_RT)
    # build the MO integrator
    if bw_mode == 'DT':
        ds.requires(rdv.medium_radiance_path_integrator_DT)
    elif bw_mode == 'DS':
        ds.requires(rdv.medium_radiance_path_integrator_NEE_DT)
    elif bw_mode == 'DRTDS':
        ds.requires(rdv.medium_radiance_path_integrator_NEE_DRTDS)
    elif bw_mode == 'DRT':
        ds.requires(rdv.medium_radiance_path_integrator_NEE_DRT)
    elif bw_mode == 'DRTQ':
        ds.requires(rdv.medium_radiance_path_integrator_NEE_DRTQ)
    elif bw_mode == 'SPS':
        ds.requires(rdv.medium_radiance_path_integrator_NEE_SPS)
    else:
        raise NotImplemented()
    ds.requires(rdv.camera_sensors, width=radiances.shape[-2], height=radiances.shape[-3], jittered=True)

    if history is None:  # First stage
        with torch.no_grad():
            # Compute history for the first state
            history_time = time.perf_counter()
            history = ds.camera.capture(ds.radiance, batch_size=512 * 512, fw_samples=256)
            print(f"[INFO] Rendered history in {time.perf_counter() - history_time}s.")

    optimization_objects = [
        create_optimization_objects(sigma_parameters, 'adam', 'none', lr, lr_decay, iterations),
    ]

    regularizers = [  # There werent used regularizers in DRT
        lambda: rdv.total_variation(ds.sigma_tensor).mean() * 0.000005
    ]

    def after_step(**stats):
        with torch.no_grad():
            # apply constraint if required!
            # sigma_parameters.clamp_(0.0, 1.0)

            stats = {**stats}
            it = stats['iteration']
            if (global_step + it + 1) % max(1, iterations // 50) == 0:
                print(str(global_step + it + 1) + "- " + (
                    '\t'.join(f"{k}:{'{a:.6f}'.format(a=v)} " for k, v in stats.items() if isinstance(v, float))))

    reconstruction_stage(
        ds=ds,
        radiances=radiances,
        loss_function=torch.nn.HuberLoss(),
        optimization_objects=optimization_objects,
        regularizers=regularizers,
        after_step=after_step,
        fw_samples=fw_samples,
        bw_samples=bw_samples,
        steps=iterations,
        batch_size=None,
        history=history,
        enhancing=enhancing_alpha
    )

    ds.forward_dependency()
    return ds.sigma_tensor.detach().clone(), history
