##################################################
# Import libraries
##################################################
from    typing      import  Callable
import  sys
import  argparse
from    shutil      import  copyfile
from    time        import  time
import  torch

from    deep_numerical.utils    import  velocity_grid
from    deep_numerical.numerical   import  distribution
from    deep_numerical.numerical.math   import  area_of_unit_sphere
from    deep_numerical.numerical.solver     import  FastSM_Boltzmann_VHS, one_step_RK4_classic

from    utils   import  ConfigLoader, sample_quantities, normalize_density


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_generation.yaml', help='Path to the configuration file.')
args = parser.parse_args()


##################################################
# Load the configuration file
##################################################
config = ConfigLoader(args.config)

# Configure computational settings
dtype   = config.dtype
device  = config.device
torch.set_default_dtype(dtype)
torch.set_default_device(device)
dtype_and_device = {'dtype': dtype, 'device': device}

# Configure dataset to be generated
PART_INIT   = config.part_init
N_REPEAT    = config.num_repeat
PART_LAST   = config.part_last
BATCH_SIZE  = config.batch_size

DELTA_T     = config.delta_t
MAX_T       = config.max_t
RES_T       = config.res_t
DATA_SIZE   = BATCH_SIZE*RES_T

DIMENSION   = config.dim
RES_V       = config.res_v
V_MAX       = config.max_v
DELTA_V     = config.delta_v

v_grid = velocity_grid(DIMENSION, RES_V, V_MAX, **dtype_and_device)
FFT_AXES:   tuple[int] = tuple(range(-(1+DIMENSION), -1))
FFT_NORM:   str  = 'forward'

V_PERTURB_RATE: float = config.v_perturb_rate
T_PERTURB_RATE: float = config.t_perturb_rate
sample_q: Callable[[int], tuple[torch.Tensor]] = \
    lambda batch_size: sample_quantities(
        DIMENSION, batch_size, v_max=V_MAX,
        v_perturb_rate=V_PERTURB_RATE, t_perturb_rate=T_PERTURB_RATE,
        **dtype_and_device,
    )


VHS_COEFF   = 1.0 / area_of_unit_sphere(DIMENSION)
DIR_NAME    = config.storage
file_dir    = config.storage / f"Boltzmann_{DIMENSION}D__coeff{VHS_COEFF:.2e}__res_t{RES_T:03d}_v{RES_V:03d}"
if not file_dir.exists():   file_dir.mkdir(parents=True)


path_log    = file_dir / config.log_name
sys.stdout  = open(path_log, "w")


copyfile(config.path, file_dir/config.path.name)
print(f"Cloned the configuration file.", flush=True)
print(f"\tDone.\n", flush=True)
config.summary()
print('\n'*2)


_LENGTH_OF_LINE = 30
for part in range(PART_INIT, PART_LAST+1):
    print('+' + '='*_LENGTH_OF_LINE + ' +')
    print(f"# part: {str(part).zfill(len(str(PART_LAST)))}")
    # Generate the initial conditions
    # NOTE: The initial conditions are shared for all collision kernels
    # NOTE: and are independently generated for each part.
    init_f: torch.Tensor = \
        0.5 * (
            distribution.maxwellian_homogeneous(v_grid, *sample_q(BATCH_SIZE))
            +
            distribution.maxwellian_homogeneous(v_grid, *sample_q(BATCH_SIZE))
        )
    init_f = normalize_density(init_f, DELTA_V)
    print(f"The initial conditions are generated (shape: {list(init_f.shape)}), which are shared for all collision kernels.", flush=True)
    
    # Compute the solutions for various collision kernels
    print(f"Computing the solutions for various collision kernels...")
    for VHS_ALPHA in config.exp_speed:
        torch.cuda.empty_cache()
        print('+' + '-'*_LENGTH_OF_LINE + ' +')
        print(f"* VHS model ((coeff, alpha) = ({VHS_COEFF:.2f}, {VHS_ALPHA:.2f}))")
        elapsed_time: float = time()
        solver = FastSM_Boltzmann_VHS(
            dimension   = DIMENSION,
            v_num_grid  = RES_V,
            v_max       = V_MAX,
            vhs_coeff   = VHS_COEFF,
            vhs_alpha   = VHS_ALPHA,
            **dtype_and_device,
        )
        FFT_CONFIG: dict[str, object] = {'s': solver.v_shape, 'dim': solver.v_axes, 'norm': FFT_NORM}

        data:   list[torch.Tensor] | torch.Tensor = []
        target: list[torch.Tensor] | torch.Tensor = []
        arr_f = init_f.clone()  # DO NOT REMOVE THE CALL OF THE METHOD `clone()`
        arr_f_fft: torch.Tensor = torch.fft.fftn(arr_f, **FFT_CONFIG)
        for cnt in range(RES_T):
            ##### 0. Save the data and target(FFT)
            q_fft   = solver.compute_fft(None, arr_f_fft)
            q       = torch.real(torch.fft.ifftn(q_fft, **FFT_CONFIG))
            data.append(arr_f.cpu())
            target.append(q.cpu())
            if cnt==RES_T-1:    break
            ##### 1. Compute the distribution at the current time step
            arr_f_fft   = solver.forward(0.0, arr_f_fft, DELTA_T, one_step_RK4_classic)
            arr_f       = torch.real(torch.fft.ifftn(arr_f_fft, **FFT_CONFIG))
            ##### 2. Save the distribution at the previous time step
            data.append(arr_f.cpu())
        elapsed_time = time()-elapsed_time
        data    = torch.stack(data, dim=1)
        target  = torch.stack(target, dim=1)
        print(f"* The shape of the data and target >>> {list(data.shape)}, {list(target.shape)}")
        print(f"* Elapsed time >>> {elapsed_time:.2f} sec")

        # Save the data
        saved_data: dict[str, object] = {
            'data':             data,
            'target':           target,
            
            'batch_size':       BATCH_SIZE,
            'max_t':            MAX_T,
            'delta_t':          DELTA_T,
            
            'res_v':            RES_V,
            'v_max':            V_MAX,
            
            'v_perturb_rate':   V_PERTURB_RATE,
            't_perturb_rate':   T_PERTURB_RATE,
            
            'vhs_coeff':    VHS_COEFF,
            'vhs_alpha':    VHS_ALPHA,
            
            'equation':     'Boltzmann',
            
            'elapsed_time': elapsed_time,
        }
        
        torch.save(saved_data, file_dir/f"alpha{VHS_ALPHA:.2f}__part{str(part).zfill(3)}.pth")
        del(data, saved_data)
        print()

    print('+' + '-'*_LENGTH_OF_LINE + ' +')
    print(f"Computation for part {part} is done.\n\n", flush=True)


print(f"All computations are done.", flush=True)


##################################################
##################################################
# End of file