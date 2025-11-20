import  sys
import  subprocess
from    pathlib             import  Path
from    ablation__config    import  *


MODEL_NAME      = "RFNO"
BASE_PATH       = Path(__file__).parent / "fno__VHS__param_only"
RUNFILE_PATH    = BASE_PATH / f"train__{MODEL_NAME.lower()}_param.py"
CUDA_INDEX      = 3
IN_CHANNELS     = 2


sys.stdout = open(BASE_PATH/f"{MODEL_NAME}{DIMENSION}D-param__ablation.txt", "w")
for max_freq in SET_MAX_FREQ:
    print(f"Running {MODEL_NAME}{DIMENSION}D with [{max_freq=}]", flush=True)
    subprocess.run(
        [
            "python", str(RUNFILE_PATH),
            
            "--seed",           str(SEED),
            "--num_epochs",     str(NUM_EPOCHS),
            "--batch_size",     str(BATCH_SIZE),
            "--learning_rate",  str(LEARNING_RATE),
            "--cuda_index",     str(CUDA_INDEX),
            
            "--path_train",     str(PATH_TRAIN),
            "--path_val",       str(PATH_VAL),
            "--path_test",      str(PATH_TEST),
            "--exp_speed",      *map(str, EXP_SPEED),
            "--resolution",     str(RESOLUTION_TRAIN),
            "--train_part",     *map(str, TRAIN_PART),
            "--val_part",       *map(str, VAL_PART),
            "--test_part",      *map(str, TEST_PART),
            "--downsample",     str(DOWNSAMPLE),
            "--num_time_steps", str(NUM_TIME_STEPS),
            
            "--dim_domain",         str(DIMENSION),
            "--max_freq",           str(max_freq),
            "--in_channels",        str(IN_CHANNELS),
            "--hidden_channels",    str(HIDDEN_CHANNELS),
            "--out_channels",       str(OUT_CHANNELS),
            "--dim_parameters",     str(DIMENSION + 1),
            "--activation_name",    ACTIVATION_NAME,
        ]
    )