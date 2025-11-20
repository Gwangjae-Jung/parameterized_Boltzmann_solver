##################################################
# Import libraries
##################################################
import  sys
from    typing              import  Union

import  time
from    pathlib             import  Path
from    copy                import  deepcopy

import  numpy       as  np
import  torch
from    torch.utils.data            import  TensorDataset, DataLoader

from    deep_numerical.utils        import  get_time_str
from    deep_numerical.utils        import  GaussianNormalizer
from    deep_numerical.utils        import  count_parameters, initialize_weights
from    deep_numerical.utils        import  relative_error
from    deep_numerical.neuralop     import  ModifiedFNO

from    train_utils         import  load_data, exponential_cosine
from    train_utils         import  LossFunctions
from    train_utils         import  params_as_constant
from    train_utils         import  CustomParser
from    train_utils         import  sec_to_hms




##################################################
# Load the configuration file
##################################################
time_str    = get_time_str(False)
args        = CustomParser()
path_script = Path(__file__).parent
DIMENSION   = len(args.n_modes)
RANDOM_SEED = args.seed

MODEL_NAME  = f"FNO{DIMENSION}D-param"
NAME_OF_DIR = f"{MODEL_NAME}__max_freq{args.n_modes[0]//2:02d}__seed{RANDOM_SEED}"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Save the config files
BATCH_SIZE:     int             = args.batch_size
NUM_EPOCHS:     int             = args.num_epochs
TRAIN_SIZE:     int             = None
VAL_SIZE:       int             = None
LEARNING_RATE:  float           = args.learning_rate
DEVICE:         torch.device    = torch.device(f"cuda:{args.cuda_index}")

RESOLUTION:     int             = args.resolution
PATH_TRAIN:     Path            = args.path_train
PATH_VAL:       Path            = args.path_val
TRAIN_PART:     tuple[int]      = args.train_part
VAL_PART:       tuple[int]      = args.val_part

NUM_TIME_STEPS: int             = args.num_time_steps
EXP_SPEED:      tuple[float]    = args.exp_speed

PATH_BASE       = path_script / MODEL_NAME
PATH_CHECKPOINT = PATH_BASE / f"checkpoint__{NAME_OF_DIR}"
PATH_BEST_MODEL = PATH_BASE / NAME_OF_DIR
PATH_BASE.mkdir(exist_ok=True)
PATH_CHECKPOINT.mkdir(exist_ok=True)
PATH_BEST_MODEL.mkdir(exist_ok=True)

sys.stdout  = open(PATH_BASE/f"{NAME_OF_DIR}.txt", "w")
print(
    f"Training the model [{MODEL_NAME}] with",
    f"* n_layers: {args.n_layers}",
    f"* max_freq: {args.n_modes[0]//2}",
    sep     = '\n',
)
print(f"* Random seed: {RANDOM_SEED}")
print(f"* The device to be used >>> {torch.cuda.get_device_name()} ({DEVICE})")
sys.stdout.flush()




##################################################
# Load data
##################################################
k_in    = 'data'
k_param = 'params'
train_data: dict[str, torch.Tensor] = {
    k_in:       None,
    k_param:    None,
}
val_data: dict[str, torch.Tensor] = {
    k_in:       None,
    k_param:    None,
}
normalizer: dict[str, GaussianNormalizer] = {
    k_in:       None,
    k_param:    None,
}


# Train data
_train_data, _train_info = load_data(PATH_TRAIN, RESOLUTION, EXP_SPEED, TRAIN_PART)
V_MAX           = _train_info['v_max']
WHERE_CLOSED    = _train_info['v_where_closed']
train_data[k_in]    = _train_data['data'][:, :NUM_TIME_STEPS]
train_data[k_param] = params_as_constant(_train_data['vhs_alpha'], DIMENSION, RESOLUTION)
TRAIN_SIZE  = int(train_data[k_in].size(0))

# Validation data
_val_data, _ = load_data(PATH_VAL, RESOLUTION, EXP_SPEED, VAL_PART)
val_data[k_in]      = _val_data['data'][:, :NUM_TIME_STEPS]
val_data[k_param]   = params_as_constant(_val_data['vhs_alpha'], DIMENSION, RESOLUTION)
VAL_SIZE    = int(val_data[k_in].shape[0])

# Normalize data
normalizer[k_in]    = GaussianNormalizer(train_data[k_in])
normalizer[k_param] = GaussianNormalizer(train_data[k_param])
train_data[k_in]    = normalizer[k_in].encode(train_data[k_in])
val_data[k_in]      = normalizer[k_in].encode(val_data[k_in])
train_data[k_param] = normalizer[k_param].encode(train_data[k_param])
val_data[k_param]   = normalizer[k_param].encode(val_data[k_param])

# Save the normalizer, which will also be used in prediction
for k in normalizer.keys():
    normalizer[k].cpu()
torch.save(normalizer, PATH_BEST_MODEL/"normalizer.pth")
for k in normalizer.keys():
    normalizer[k].to(DEVICE)

print(f"The size of the training dataset >>>", TRAIN_SIZE, sep=' ')
print(f"The shape of the training dataset >>>", train_data[k_in].shape, sep=' ')
print(f"The size of the validation dataset >>>", VAL_SIZE, sep=' ')
print(f"The shape of the validation dataset >>>", val_data[k_in].shape, sep=' ')
sys.stdout.flush()




##################################################
# Instantiate dataloaders
##################################################
# Train data
train_dataset   = TensorDataset(train_data[k_in], train_data[k_param])
train_loader    = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Validation data
val_dataset   = TensorDataset(val_data[k_in], val_data[k_param])
val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)




##################################################
# Instantiate models, loss functions, optimizers, and learning-rate schedulers
##################################################
model = ModifiedFNO(**args.config_model('fno')).to(DEVICE)
initialize_weights(model, "xavier normal")

lf = LossFunctions(dimension=DIMENSION, resolution=RESOLUTION, v_max=V_MAX, v_where_closed=WHERE_CLOSED, device=DEVICE)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
lr_lambda = exponential_cosine(period=args.lr_period, half_life=float('inf'))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

print(f"The number of the parameters in the models\n>>> {count_parameters(model)}")
sys.stdout.flush()




##################################################
# Train the model
##################################################
train_history: dict[str, Union[list, float]] = {
    'train_rel_error':      [],
    'train_rel_error_cons': [],
    'val_rel_error':        [],
    'val_rel_error_cons':   [],
    'train_time':           0.0,
}

best_val_rel_error:         float   = None
best_val_rel_error_cons:    float   = None
best_model:         ModifiedFNO     = None
best_model_idx:     int             = None

USED_TIME_STEPS:    int = 2

_denominator_train  = TRAIN_SIZE * USED_TIME_STEPS * (NUM_TIME_STEPS - USED_TIME_STEPS)
_denominator_val    = VAL_SIZE * USED_TIME_STEPS * (NUM_TIME_STEPS - USED_TIME_STEPS)
arr_coeff_cons = 0.5 * (1.0-torch.pow(0.9, torch.arange(NUM_EPOCHS, device=DEVICE)))
for k in train_data.keys():
    normalizer[k].to(DEVICE)

elapsed_time = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    # NOTE: Train
    model.train()
    coeff_cons = arr_coeff_cons[epoch-1]
    _train_time = time.time()
    train_rel_error:        float   = 0
    train_rel_error_cons:   float   = 0
    for traj, params in train_loader:
        traj:   torch.Tensor = traj.to(DEVICE)
        params: torch.Tensor = params.to(DEVICE)
        
        for idx in range(NUM_TIME_STEPS - USED_TIME_STEPS):
            # Time-marching (in the scaled space)
            data_prev = traj[:, idx]
            data_curr = traj[:, idx+1]
            data_next = traj[:, idx+2]
            pred_curr = model.forward(data_prev, params)
            pred_next = model.forward(pred_curr, params)
            # Descale data
            pred_curr = normalizer[k_in].decode(pred_curr)
            pred_next = normalizer[k_in].decode(pred_next)
            data_curr = normalizer[k_in].decode(data_curr)
            data_next = normalizer[k_in].decode(data_next)
            # Loss 1 - Data-driven loss
            rel_error__data = \
                relative_error(pred_curr, data_curr, p=2).sum() + \
                relative_error(pred_next, data_next, p=2).sum()
            # Loss 2 - Conservation loss
            rel_error__cons = lf.compute_loss_cons(
                [pred_curr, pred_next],
                [data_curr, data_next],
            )
            # Compute the total loss
            loss: torch.Tensor = \
                rel_error__data + coeff_cons * rel_error__cons
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Backup
            train_rel_error         += rel_error__data.item()
            train_rel_error_cons    += rel_error__cons.item()
        
    scheduler.step()
    _train_time = time.time() - _train_time
    train_rel_error         /= _denominator_train
    train_rel_error_cons    /= _denominator_train
    train_history['train_time'] += _train_time
    train_history['train_rel_error'].append(train_rel_error)
    train_history['train_rel_error_cons'].append(train_rel_error_cons)
    
    # NOTE: Validation
    model.eval()
    val_rel_error:      float   = 0
    val_rel_error_cons: float   = 0
    with torch.no_grad():
        for traj, params in val_loader:
            traj:   torch.Tensor = traj.to(DEVICE)
            params: torch.Tensor = params.to(DEVICE)
            
            for idx in range(NUM_TIME_STEPS - USED_TIME_STEPS):
                # Time-marching (in the scaled space)
                data_prev = traj[:, idx]
                data_curr = traj[:, idx+1]
                data_next = traj[:, idx+2]
                pred_curr = model.forward(data_prev, params)
                pred_next = model.forward(pred_curr, params)
                # Descale data
                pred_curr = normalizer[k_in].decode(pred_curr)
                pred_next = normalizer[k_in].decode(pred_next)
                data_curr = normalizer[k_in].decode(data_curr)
                data_next = normalizer[k_in].decode(data_next)
                # Loss 1 - Data-driven loss
                rel_error__data = \
                    relative_error(pred_curr, data_curr, p=2).sum() + \
                    relative_error(pred_next, data_next, p=2).sum()
                # Loss 2 - Conservation loss
                rel_error__cons = lf.compute_loss_cons(
                    [pred_curr, pred_next],
                    [data_curr, data_next],
                )
                # Backup
                val_rel_error       += rel_error__data.item()
                val_rel_error_cons  += rel_error__cons.item()
            
    val_rel_error       /= _denominator_val
    val_rel_error_cons  /= _denominator_val
    train_history['val_rel_error'].append(val_rel_error)
    train_history['val_rel_error_cons'].append(val_rel_error_cons)
    
    # NOTE: Save the best model
    if (
            (best_val_rel_error is None or best_val_rel_error_cons is None) or 
            (best_val_rel_error+best_val_rel_error_cons > val_rel_error+val_rel_error_cons)
        ):
        best_val_rel_error      = val_rel_error
        best_val_rel_error_cons = val_rel_error_cons
        best_model      = deepcopy(model)
        best_model_idx  = epoch-1
        
    # NOTE: Report
    print()
    print(f"[ Epoch {epoch}/{NUM_EPOCHS} ] {train_history['train_time']/epoch:.1f}s/epoch")
    print(f"- coeff_cons: {coeff_cons:.4e} , lr: {scheduler.get_last_lr()[0]:.4e}")
    print(
        "# Best model:",
        f"epoch {best_model_idx+1}",
        f"({train_history['val_rel_error'][best_model_idx]:.4e}",
        f"|",
        f"{train_history['val_rel_error_cons'][best_model_idx]:.4e})",
        sep=' ',
    )
    print(
        "* [train] Relative error:",
        f"{train_history['train_rel_error'][-1]:.4e}",
        f"|",
        f"{train_history['train_rel_error_cons'][-1]:.4e}",
        sep=' ',
    )
    print(
        "* [valid] Relative error:",
        f"{train_history['val_rel_error'][-1]:.4e}",
        f"|",
        f"{train_history['val_rel_error_cons'][-1]:.4e}",
        sep=' ',
    )
    print('\n', flush=True)
        
    if (
            best_val_rel_error < 1e-5 and
            train_history['val_rel_error_cons'][best_model_idx] < 1e-5
        ):
        print(f"Early stopping at epoch {epoch}.", flush=True)
        break
    
    if epoch%10==0:
        # Save the model
        __appendix__ = f"epoch{str(epoch).zfill(len(str(NUM_EPOCHS)))}"
        __name_checkpoint = f"state_dict__{__appendix__}.pth"
        model.cpu()
        best_model.cpu()
        torch.save(model.state_dict(), PATH_CHECKPOINT/__name_checkpoint)
        torch.save(best_model.state_dict(), PATH_CHECKPOINT/f"best__{__name_checkpoint}")
        model.to(DEVICE)
        best_model.to(DEVICE)
        torch.save(train_history, PATH_CHECKPOINT/f"train_history__{__appendix__}.pth")
        
elapsed_time = time.time() - elapsed_time
print(f"Elapsed time: {sec_to_hms(int(elapsed_time))}")

model = best_model




##################################################
# Save the model and the train history
##################################################
model.cpu()
for k in train_data.keys():
    normalizer[k].cpu()

# Save the model
torch.save(model.state_dict(), PATH_BEST_MODEL/"state_dict.pth")

# Save the history
torch.save(train_history, PATH_BEST_MODEL/"train_history.pth")

# Clear the GPU memory
torch.cuda.empty_cache()

# Done
print("Done.", flush=True)




##################################################
##################################################
# End of file