DIMENSION       = 2
SET_MAX_FREQ    = [8, 6, 10]
SET_N_MODES     = [tuple((2*max_freq+1 for _ in range(DIMENSION))) for max_freq in SET_MAX_FREQ]
SET_N_LAYERS    = [4, 3, 5, 6]


#####< Experiment parameters >#####
SEED            = 0
NUM_EPOCHS      = 1000
BATCH_SIZE      = 1024
LEARNING_RATE   = 1e-4


#####< Dataset parameters >#####
RESOLUTION_TRAIN    = 32
RESOLUTION_VAL      = 32
RESOLUTION_TEST     = 32
PATH_TRAIN          = \
    rf"/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE/parameterized_Boltzmann_solver/datasets_kinetic/Boltzmann_2D/biMaxwellian/coeff1.59e-01/res{str(RESOLUTION_TRAIN).zfill(3)}"
PATH_VAL            = \
    rf"/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE/parameterized_Boltzmann_solver/datasets_kinetic/Boltzmann_2D/biMaxwellian/coeff1.59e-01/res{str(RESOLUTION_VAL).zfill(3)}"
PATH_TEST           = \
    rf"/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE/parameterized_Boltzmann_solver/datasets_kinetic/Boltzmann_2D/biMaxwellian/coeff1.59e-01/res{str(RESOLUTION_TEST).zfill(3)}"
EXP_SPEED           = [0.0, 0.25, 0.5, 0.75, 1.0]
TRAIN_PART          = list(range(1, 11))
VAL_PART            = list(range(11, 13))
TEST_PART           = list(range(1, 10))
DOWNSAMPLE          = 1
NUM_TIME_STEPS      = 51


#####< Model parameters >#####
HIDDEN_CHANNELS     = 32
OUT_CHANNELS        = 1

LIFT_LAYER          = [128, 128]
N_LAYERS            = 4
PROJECT_LAYER       = [128, 128]

ACTIVATION_NAME     = "relu"
ACTIVATION_KWARGS   = {}
POS_ENC             = True

DELTA_T             = 0.1