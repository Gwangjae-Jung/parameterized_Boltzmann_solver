##################################################
# Import libraries
##################################################
from    typing      import  Callable
from    itertools   import  product

from    time        import  time
import  yaml

import  torch
from    helper.kinetic_distribtutions   import  *

from    pathlib             import  Path
root_dir    = r"/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE"
path_root   = Path(root_dir)
"""Directory `/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE`"""
path_lib    = path_root / "core"

from    sys     import  path
path.append( str(path_lib) )
from    core     import  utils
from    core.numerical   import  distribution
from    core.numerical.solvers     import  FastSM_Boltzmann_VSS

from    shutil  import  copyfile




##################################################
# Load the configuration file
##################################################
print(f"Loading the configuration file...")
NAME_OF_CONFIG_FILE = "config_generation.yaml"
with open(NAME_OF_CONFIG_FILE) as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
print(f"\tDone.")


# Configure computational settings
config_computation = config['computation']
dtype:  torch.dtype     = getattr(torch, config_computation['dtype'])
device: torch.device    = torch.device(config_computation['device'])
dtype_and_device = {'dtype': dtype, 'device': device}
__dtype_str = str(dtype).split('.')[-1]


# Configure dataset to be generated
config_dataset = config['dataset']
PART_INIT:  int     = config_dataset['part_init']
N_REPEAT:   int     = config_dataset['n_repeat']
PART_LAST:  int     = PART_INIT + N_REPEAT - 1
NUM_INST:   int     = config_dataset['n_inst']

DELTA_T:    float   = config_dataset['delta_t']
MAX_T:      float   = config_dataset['max_t']
NUM_T:      int     = 1 + int(MAX_T/DELTA_T + 0.1)
DATA_SIZE:  int     = NUM_INST * NUM_T

DIMENSION:  int     = config_dataset['dimension']
RESOLUTION: int     = config_dataset['resolution']
V_MAX:      float   = 3.0/utils.LAMBDA
DELTA_V:    float   = (2*V_MAX) / RESOLUTION
V_WHERE_CLOSED: str = config_dataset['v_where_closed']

v_grid = utils.velocity_grid(
    DIMENSION, RESOLUTION, V_MAX,
    where_closed=V_WHERE_CLOSED,
    **dtype_and_device
)

FFT_AXES:   tuple[int] = tuple(range(-(1+DIMENSION), -1))
FFT_NORM:   str  = 'forward'

V_PERTURB_RATE: float = config_dataset['v_perturb_rate']
sample_q: Callable[[int], tuple[torch.Tensor]] = \
    lambda batch_size: sample_quantities(
        DIMENSION, batch_size, v_max=V_MAX,
        v_perturb_rate=V_PERTURB_RATE, 
        **dtype_and_device,
    )




##################################################
# Computation begins
##################################################
VSS_COEFF   = 1.0 / utils.area_of_unit_sphere(DIMENSION)

DIR_NAME    = config_dataset['dir_name']
_res_str    = f"res{str(RESOLUTION).zfill(3)}"
file_dir    = path_root / DIR_NAME / f"Boltzmann_{DIMENSION}D_VSS" / "biMaxwellian" / f"coeff{VSS_COEFF:.2e}" / _res_str
if not Path.exists(file_dir):
    Path.mkdir(file_dir, parents=True)    
print(f"Saving the configuration file...")
copyfile(NAME_OF_CONFIG_FILE, file_dir/NAME_OF_CONFIG_FILE)
print(f"\tDone.")


for part in range(PART_INIT, PART_LAST+1):
    print('+' + '='*30 + ' +')
    print(f"# part: {str(part).zfill(len(str(PART_LAST)))}")
    # Generate the initial conditions
    # The initial conditions are shared for all collision kernels
    # and are independently generated for each part.
    ##################################################
    # (Range of edition 1)
    # NOTE: Edit this part to change the initial conditions.
    init_f: torch.Tensor = \
        0.5 * (
            distribution.maxwellian_homogeneous(v_grid, *sample_q(NUM_INST))
            +
            distribution.maxwellian_homogeneous(v_grid, *sample_q(NUM_INST))
        )
    # NOTE: The range of edition 1 ends here.
    ##################################################
    init_f = normalize_density(init_f, DELTA_V)
    print(f"The initial conditions are generated (shape: {list(init_f.shape)}), which are shared for all collision kernels.")
    
    
    # Compute the solutions for various collision kernels
    print(f"Computing the solutions for various collision kernels...")
    for VSS_EXP_SPEED, VSS_EXP_ANGLE in product(config_dataset['exp_speed'], config_dataset['exp_angle']):
        torch.cuda.empty_cache()
        print('+' + '-'*30 + ' +')
        print(f"* VSS model (coeff, exp_speed, exp_angle): {VSS_COEFF:.2f}, {VSS_EXP_SPEED:.2f}, {VSS_EXP_ANGLE:.2f}")
        elapsed_time: float = time()
        ##################################################
        # (Range of edition 2)
        # NOTE: Edit this part to change the solver.
        solver = FastSM_Boltzmann_VSS(
            dimension   = DIMENSION,
            v_num_grid  = RESOLUTION,
            v_max       = V_MAX,
            
            vss_coeff       = VSS_COEFF,
            vss_exp_speed   = VSS_EXP_SPEED,
            vss_exp_angle   = VSS_EXP_ANGLE,
            
            quad_order_uniform  = RESOLUTION,
            quad_order_legendre = int(0.7*RESOLUTION),

            dtype   = dtype,
        ).to(device)
        # NOTE: The range of edition 2 ends here.
        ##################################################
        FFT_CONFIG: dict[str, object] = {'s': solver.v_shape, 'dim': solver.v_axes, 'norm': FFT_NORM}


        data:   list[torch.Tensor]  = []
        arr_f = init_f.clone()  # DO NOT REMOVE THE CALL OF THE METHOD `clone()`
        arr_f_fft: torch.Tensor = torch.fft.fftn(arr_f, **FFT_CONFIG)

        for cnt in range(NUM_T):
            ##### 1. Save the distribution at the previous time step
            data.append(arr_f.cpu())
            ##### 2. Compute the distribution at the current time step
            if cnt<NUM_T-1:
                arr_f_fft = solver.forward(0.0, arr_f_fft, DELTA_T, utils.one_step_RK4_classic)
                arr_f = torch.real(torch.fft.ifftn(arr_f_fft, **FFT_CONFIG))
        data:   torch.Tensor    = torch.stack(data, dim=1)
        elapsed_time = time() - elapsed_time
        print(f"* The shape of the data >>> {data.shape}")
        print(f"* Elapsed time >>> {elapsed_time:.2f} sec")

        # Save the data
        saved_data: dict[str, object] = {
            'data':             data,
            
            'n_init':           NUM_INST,
            
            'max_t':            MAX_T,
            'delta_t':          DELTA_T,
            
            'resolution':       RESOLUTION,
            'v_max':            V_MAX,
            'v_where_closed':   V_WHERE_CLOSED,
            
            'v_perturb_rate':   V_PERTURB_RATE,
            
            'vss_coeff':        VSS_COEFF,
            'vss_exp_speed':    VSS_EXP_SPEED,
            'vss_exp_angle':    VSS_EXP_ANGLE,

            'equation':     'Boltzmann',
            'dtype_str':    __dtype_str,
            
            'elapsed_time': elapsed_time,
        }
        
        # file_name = f"{_res_str}__alpha{float(VHS_ALPHA):.1e}__part{str(part).zfill(len(str(PART_LAST)))}.pth"
        file_name = f"{_res_str}__alpha{float(VSS_EXP_SPEED):.1e}__beta{float(VSS_EXP_ANGLE):.1e}__part{str(part).zfill(3)}.pth"
        print(f"Saving data as follows:")
        print(f"* Directory: {file_dir}")
        print(f"* File name: {file_name}")
        torch.save(saved_data, file_dir/file_name)
        del(data, saved_data)
        print('\n'*2)

    print(f"Computation for part {part} is done.")
    print('\n'*2)


print(f"All computations are done.")




##################################################
##################################################
# End of file