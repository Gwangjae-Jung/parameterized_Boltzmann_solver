import  argparse

import  sys
import  subprocess
from    pathlib             import  Path
from    ablation__config    import  *


MODEL_NAME      = "PFNO"
BASE_PATH       = Path(__file__).parent / "fno__VHS__param_only"
RUNFILE_PATH    = BASE_PATH / f"train__{MODEL_NAME.lower()}_param.py"
IN_CHANNELS     = 1


parser = argparse.ArgumentParser()
parser.add_argument("--cuda_index", type=int, help="The CUDA device to be used.")
args = parser.parse_args()


sys.stdout = open(BASE_PATH/f"{MODEL_NAME}{DIMENSION}D-param__ablation.txt", "w")
for n_modes in SET_N_MODES:
    print(f"Running {MODEL_NAME}{DIMENSION}D with [{n_modes=}]", flush=True)
    subprocess.run(
        [
            "python", str(RUNFILE_PATH),
            
            "--seed",           str(SEED),
            "--num_epochs",     str(NUM_EPOCHS),
            "--batch_size",     str(BATCH_SIZE),
            "--learning_rate",  str(LEARNING_RATE),
            "--cuda_index",     str(args.cuda_index),
            
            "--path_train",     str(PATH_TRAIN),
            "--path_val",       str(PATH_VAL),
            "--path_test",      str(None),
            "--exp_speed",      *map(str, EXP_SPEED),
            "--resolution",     str(RESOLUTION_TRAIN),
            "--train_part",     *map(str, TRAIN_PART),
            "--val_part",       *map(str, VAL_PART),
            "--test_part",      *map(str, TEST_PART),
            "--downsample",     str(DOWNSAMPLE),
            "--num_time_steps", str(NUM_TIME_STEPS),
            
            "--n_modes",            *map(str, n_modes),
            "--in_channels",        str(IN_CHANNELS),
            "--hidden_channels",    str(HIDDEN_CHANNELS),
            "--out_channels",       str(OUT_CHANNELS),
            "--dim_parameters",     str(1),
            "--activation_name",    ACTIVATION_NAME,
        ]
    )