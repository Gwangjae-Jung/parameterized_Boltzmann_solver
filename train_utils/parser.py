from    typing      import  *


__all__: list[str] = ['CustomParser']


class CustomParser():
    def __init__(self) -> Self:
        import  argparse
        
        parser = argparse.ArgumentParser()
        group_experiment = parser.add_argument_group("Experiment parameters")
        group_dataset = parser.add_argument_group("Dataset parameters")
        group_model = parser.add_argument_group("Model parameters")
        
        #####< Experiment parameters >#####
        group_experiment.add_argument(
            '--seed', type=int,
            help="The random seed to be used.",
        )
        group_experiment.add_argument(
            '--num_epochs', type=int,
            help="The number of epochs to be used in training.",
        )
        group_experiment.add_argument(
            '--batch_size', type=int,
            help="The batch size to be used in training/validation/testing.",
        )
        group_experiment.add_argument(
            '--learning_rate', type=float,
            help="The learning rate to be used in the optimizer.",
        )
        group_experiment.add_argument(
            '--lr_period', type=int,
            help="The period of learning rate decay.",
            default=20,
        )
        group_experiment.add_argument(
            '--cuda_index', type=int,
            help="The CUDA device index to be used.",
        )
        
        #####< Dataset parameters >#####
        group_dataset.add_argument(
            '--resolution', type=int, 
            help="The resolution of the data in each dimension.",
        )
        group_dataset.add_argument(
            '--path_train', type=str, 
            help="The path to the training dataset.",
        )
        group_dataset.add_argument(
            '--path_val', type=str, 
            help="The path to the validation dataset.",
        )
        group_dataset.add_argument(
            '--path_test', type=str, 
            help="The path to the testing dataset.",
        )
        group_dataset.add_argument(
            '--exp_speed', type=float, 
            help="The power of the speed in the collision kernels.",
            nargs='+',
        )
        group_dataset.add_argument(
            '--train_part', type=int,
            help="The parts of the training dataset to be used.",
            nargs='+',
        )
        group_dataset.add_argument(
            '--val_part', type=int,
            help="The parts of the validation dataset to be used.",
            nargs='+',
        )
        group_dataset.add_argument(
            '--test_part', type=int,
            help="The parts of the testing dataset to be used.",
            nargs='+',
        )
        group_dataset.add_argument(
            '--downsample', type=int,
            help="The downsampling rate to be applied to the data.",
        )
        group_dataset.add_argument(
            '--num_time_steps', type=int,
            help="The number of time steps in each sample.",
        )
        
        #####< Model parameters >#####
        group_model.add_argument(
            '--dim_domain', type=int,
            help="The dimension of the spatial domain.",
        )
        group_model.add_argument(
            '--max_freq', type=int,
            help="The maximum frequency to be considered in the model.",
        )
        group_model.add_argument(
            '--n_modes', type=int,
            help="The number of the Fourier modes saved in each dimension.",
            nargs='+',
        )
        group_model.add_argument(
            '--in_channels', type=int,
            help="The number of the channels in the input data. This need not be the same to 'dim'.",
        )
        group_model.add_argument(
            '--hidden_channels', type=int,
            help="The number of the channels in the hidden layers. Detailed explanation on the usage of 'hidden_channels' can be found in the documentations of the models to be used.",
        )
        group_model.add_argument(
            '--out_channels', type=int,
            help="The number of the channels in the output data.",
        )
        group_model.add_argument(
            '--dim_parameters', type=int,
            help="The dimension of the parameter space (only for parametric models).",
        )
        
        group_model.add_argument(
            '--lift_layer', type=int,
            help="The number of intermediate layers comprising the lifting layer.",
            nargs='+', default=[128, 128],
        )
        group_model.add_argument(
            '--n_layers', type=int,
            help="The number of the hidden layers.",
            default = 4,
        )
        group_model.add_argument(
            '--project_layer', type=int,
            help="The number of intermediate layers comprising the project layer.",
            nargs='+', default=[128, 128],
        )
        
        group_model.add_argument(
            '--activation_name', type=str,
            help="The name of the activation function to be used.",
            default = "relu",
        )
        group_model.add_argument(
            '--delta_t', type=float,
            help="The parameter to be used in the temporal simulation models.",
        )
        
        group_model.add_argument(
            '--pos_enc', action='store_true',
            help="Whether to use positional encoding or not.",
            default=True,
        )
        
        #####< Done >#####
        self.args = parser.parse_args()
        return


    #####< Properties: Dataset parameters >#####
    @property
    def resolution(self) -> int:            return self.args.resolution
    @property
    def path_train(self) -> str:            return self.args.path_train
    @property
    def path_val(self) -> str:              return self.args.path_val
    @property
    def path_test(self) -> str:             return self.args.path_test
    @property
    def exp_speed(self) -> tuple[float]:    return tuple(self.args.exp_speed)
    @property
    def train_part(self) -> tuple[int]:     return tuple(self.args.train_part)
    @property
    def val_part(self) -> tuple[int]:       return tuple(self.args.val_part)
    @property
    def test_part(self) -> tuple[int]:      return tuple(self.args.test_part)
    @property
    def downsample(self) -> int:            return self.args.downsample
    @property
    def num_time_steps(self) -> int:        return self.args.num_time_steps
    
    #####< Properties: Experiment parameters >#####
    @property
    def seed(self) -> int:              return self.args.seed
    @property
    def num_epochs(self) -> int:        return self.args.num_epochs
    @property
    def batch_size(self) -> int:        return self.args.batch_size
    @property
    def learning_rate(self) -> float:   return self.args.learning_rate
    @property
    def lr_period(self) -> int:         return self.args.lr_period
    @property
    def cuda_index(self) -> int:        return self.args.cuda_index
    
    #####< Properties: Model parameters >#####
    @property
    def dim_domain(self) -> int:            return self.args.dim_domain
    @property
    def max_freq(self) -> int:              return self.args.max_freq
    @property
    def n_modes(self) -> list[int]:         return list(self.args.n_modes)
    @property
    def in_channels(self) -> int:           return self.args.in_channels
    @property
    def hidden_channels(self) -> int:       return self.args.hidden_channels
    @property
    def out_channels(self) -> int:          return self.args.out_channels
    @property
    def dim_parameters(self) -> int:        return self.args.dim_parameters
    @property
    def lift_layer(self) -> list[int]:      return list(self.args.lift_layer)
    @property
    def n_layers(self) -> int:              return self.args.n_layers
    @property
    def project_layer(self) -> list[int]:   return list(self.args.project_layer)
    @property
    def activation_name(self) -> str:       return self.args.activation_name
    @property
    def delta_t(self) -> float:             return self.args.delta_t
    @property
    def pos_enc(self) -> bool:              return self.args.pos_enc
    
    
    def config_model(self, model_name: str) -> dict[str, Any]:
        base_config = {
            'in_channels':      self.in_channels,
            'hidden_channels':  self.hidden_channels,
            'out_channels':     self.out_channels,
            'lift_layer':       self.lift_layer,
            'n_layers':         self.n_layers,
            'project_layer':    self.project_layer,
            'activation_name':  self.activation_name,
            'bias':             False,
        }
        if model_name.lower() == 'fno':
            base_config['n_modes']          = self.n_modes
        elif model_name.lower() == 'rfno':
            base_config['dim_domain']       = self.dim_domain
            base_config['max_freq']         = self.max_freq
        elif model_name.lower() == 'pfno':
            base_config['n_modes']          = self.n_modes
            base_config['dim_parameters']   = self.dim_parameters
        elif model_name.lower() == 'prfno':
            base_config['dim_domain']       = self.dim_domain
            base_config['max_freq']         = self.max_freq
            base_config['dim_parameters']   = self.dim_parameters
        else:
            raise ValueError(f"[CustomParser.config_model] The model name '{model_name}' is not recognized.")
        return base_config
    

##################################################
##################################################
# End of file