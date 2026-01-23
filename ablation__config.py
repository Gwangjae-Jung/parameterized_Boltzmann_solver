#####< Modifiable options for ablation study >#####
DIMENSION       = 3
SET_MAX_FREQ    = [8, 6, 10]
SET_N_MODES     = [tuple((2*max_freq+1 for _ in range(DIMENSION))) for max_freq in SET_MAX_FREQ]
SET_N_LAYERS    = [4, 3, 5, 6]


#####< Experiment parameters >#####
SEED            = 2
NUM_EPOCHS      = 1000
BATCH_SIZE      = 1024 if DIMENSION==2 else 512
LEARNING_RATE   = 1e-4


#####< Dataset parameters >#####
def dimention_to_coefficient() -> float:
    from    math    import  pi
    if DIMENSION==2:    return 1/(2*pi)
    elif DIMENSION==3:  return 1/(4*pi)
    else:               raise NotImplementedError
coeff = dimention_to_coefficient()
RESOLUTION_TRAIN    = 32
RESOLUTION_VAL      = 32
RESOLUTION_TEST     = 64
PATH_TRAIN          = \
    rf"/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE/parameterized_Boltzmann_solver/datasets/train/Boltzmann_{DIMENSION}D__coeff{coeff:.2e}__res_t051_v{str(RESOLUTION_TRAIN).zfill(3)}"
PATH_VAL            = \
    rf"/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE/parameterized_Boltzmann_solver/datasets/validation/Boltzmann_{DIMENSION}D__coeff{coeff:.2e}__res_t051_v{str(RESOLUTION_VAL).zfill(3)}"
EXP_SPEED           = [0.0, 0.25, 0.5, 0.75, 1.0]
TRAIN_PART          = list(range(1, 11))    # 1~10
VAL_PART            = list(range(11, 13))   # 11~12
TEST_PART           = [1]                   # Not used in training
DOWNSAMPLE          = 1
NUM_TIME_STEPS      = 51


#####< Model parameters >#####
# `IN_CHANNELS` should be specified for each model
HIDDEN_CHANNELS     = 32
OUT_CHANNELS        = 1

LIFT_LAYER          = [128, 128]
N_LAYERS            = 4
PROJECT_LAYER       = [128, 128]

ACTIVATION_NAME     = "relu"
ACTIVATION_KWARGS   = {}
POS_ENC             = True

DELTA_T             = 0.1   # NOT IN USE, BUT DEFINED FOR COMPATIBILITY