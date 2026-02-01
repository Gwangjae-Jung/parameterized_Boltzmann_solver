##################################################
# Import libraries
##################################################
import  sys
import  argparse
from    time        import  time
import  torch

from    deep_numerical.utils    import  velocity_grid
from    deep_numerical.numerical.math   import  area_of_unit_sphere
from    deep_numerical.numerical.solver     import  FastSM_Boltzmann_VHS, one_step_RK4_classic

from    utils   import  ConfigLoader


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_generation__load.yaml', help='Path to the configuration file.')
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


VHS_COEFF   = 1.0 / area_of_unit_sphere(DIMENSION)
DIR_NAME    = config.storage
file_dir    = config.storage / f"Boltzmann_{DIMENSION}D__coeff{VHS_COEFF:.2e}__res_t{RES_T:03d}_v{RES_V:03d}"
if not file_dir.exists():   raise FileNotFoundError(f"The directory {file_dir} does not exist. Please create it first.")


path_log    = file_dir / config.log_name
sys.stdout  = open(path_log, "w")


print(f"Starting to generate trajectories with", flush=True)
print(f"- (input) the initial conditions from the files 'alpha0.00__part*.pth'.", flush=True)
print(f"- (output) 'alpha' is given below:", flush=True)
for alpha in config.exp_speed:
    print(f"  * {alpha:.2f}", flush=True)


_LENGTH_OF_LINE = 30
for part in range(PART_INIT, PART_LAST+1):
    print('+' + '='*_LENGTH_OF_LINE + ' +', flush=True)
    print(f"# part: {str(part).zfill(len(str(PART_LAST)))}", flush=True)
    # Generate the initial conditions
    # NOTE: The initial conditions are shared for all collision kernels
    # NOTE: and are independently generated for each part.
    file: torch.Tensor = torch.load(file_dir/f"alpha0.00__part{str(part).zfill(3)}.pth", weights_only=False)
    init_f: torch.Tensor = file['data'][:, 0, ...]    # (batch_size, *x, *v, channel)
    init_f = init_f.to(device)
    V_PERTURB_RATE: float = file['v_perturb_rate']
    T_PERTURB_RATE: float = file['t_perturb_rate']
    print(f"The initial conditions are loaded (shape: {list(init_f.shape)}), which are shared for all collision kernels.", flush=True)
    
    # Compute the solutions for various collision kernels
    print(f"Computing the solutions for various collision kernels...", flush=True)
    for VHS_ALPHA in config.exp_speed:
        torch.cuda.empty_cache()
        print('+' + '-'*_LENGTH_OF_LINE + ' +', flush=True)
        print(f"* VHS model ((coeff, alpha) = ({VHS_COEFF:.2f}, {VHS_ALPHA:.2f}))", flush=True)
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

        data: list[torch.Tensor] | torch.Tensor = [init_f.cpu()]
        arr_f = init_f.clone()  # DO NOT REMOVE THE CALL OF THE METHOD `clone()`
        arr_f_fft: torch.Tensor = torch.fft.fftn(arr_f, **FFT_CONFIG)
        for cnt in range(RES_T-1):
            ##### 1. Compute the distribution at the current time step
            arr_f_fft   = solver.forward(0.0, arr_f_fft, DELTA_T, one_step_RK4_classic)
            arr_f       = torch.real(torch.fft.ifftn(arr_f_fft, **FFT_CONFIG))
            ##### 2. Save the distribution at the previous time step
            data.append(arr_f.cpu())
        elapsed_time = time()-elapsed_time
        data = torch.stack(data, dim=1)
        print(f"* The shape of the data >>> {list(data.shape)}", flush=True)
        print(f"* Elapsed time >>> {elapsed_time:.2f} sec", flush=True)

        # Save the data
        saved_data: dict[str, object] = {
            'data':             data,
            
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
        print(flush=True)

    print('+' + '-'*_LENGTH_OF_LINE + ' +', flush=True)
    print(f"Computation for part {part} is done.\n\n", flush=True)


print(f"All computations are done.", flush=True)


##################################################
##################################################
# End of file