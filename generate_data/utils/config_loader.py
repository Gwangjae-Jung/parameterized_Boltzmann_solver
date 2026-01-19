import  torch
from    pathlib     import  Path
from    typing      import  Any, Literal, Self, Union
import  yaml


class ConfigLoader():
    def __init__(self, path: Union[str, Path]) -> Self:
        self.__path = Path(path)
        if not self.__path.is_file():
            import  warnings
            warnings.warn(f"The configuration file '{self.__path}' does not exist.")
            return
        self.__config: dict[Literal['computation', 'dataset'], dict[str, Any]]
        with open(self.__path, 'r') as file:
            try:
                self.__config = yaml.load(file, Loader=yaml.FullLoader)
                assert 'computation' in self.__config.keys()
                assert 'dataset' in self.__config.keys()
            except:
                import  warnings
                warnings.warn(f"The configuration file '{self.__path}' is not properly formatted.")
                self.__config = dict()
                return
        self.load_config()
        return


    def load_config(self) -> None:
        from    datetime    import  datetime
        from    deep_numerical.numerical.solver.constants   import  LAMBDA
        computation = self.__config['computation']
        dataset     = self.__config['dataset']
        
        self.__dtype:       torch.dtype     = getattr(torch, computation['dtype'])
        self.__device:      torch.device    = torch.device(computation['device'])
        
        self.__part_init:   int = dataset['part_init']
        self.__num_repeat:  int = dataset['num_repeat']
        self.__part_last:   int = self.__part_init+self.__num_repeat-1
        self.__batch_size:  int = dataset['batch_size']
        
        self.__delta_t:     float   = dataset['delta_t']
        self.__max_t:       float   = dataset['max_t']
        
        self.__dim:         int     = dataset['dim']
        self.__res_v:       int     = dataset['res_v']
        self.__max_v:       float   = dataset['supp_v']/LAMBDA
        
        self.__v_perturb_rate: float = dataset['v_perturb_rate']
        self.__t_perturb_rate: float = dataset['t_perturb_rate']
        
        self.__exp_speed:   list[float] = dataset['exp_speed']
        self.__exp_angle:   list[float] = dataset['exp_angle']
        
        self.__storage      = Path(dataset['storage'])
        self.__storage.mkdir(parents=True, exist_ok=True)
        self.__log_name     = f"log__{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        return
    
    
    def summary(self) -> None:
        _length_of_line = 50
        print("="*_length_of_line)
        print("Configuration Summary:")
        print(f"* dtype:          {self.dtype}")
        print(f"* device:         {self.device}")
        print(f"* part init:      {self.part_init}")
        print(f"* num repeat:     {self.num_repeat}")
        print(f"* part last:      {self.part_last}")
        print(f"* batch size:     {self.batch_size}")
        print(f"* delta t:        {self.delta_t:.2f}")
        print(f"* max t:          {self.max_t:.2f}")
        print(f"* res t:          {self.res_t}")
        print(f"* dimension:      {self.dim}")
        print(f"* res v:          {self.res_v}")
        print(f"* max v:          {self.max_v:.2f}")
        print(f"* delta v:        {self.delta_v:.2f}")
        print(f"* v perturb rate: {self.v_perturb_rate}")
        print(f"* t perturb rate: {self.t_perturb_rate}")
        print(f"* exp speed:      {self.exp_speed}")
        print(f"* exp angle:      {self.exp_angle}")
        print(f"* storage:        {self.storage}")
        print(f"* log name:       {self.log_name}")
        print("="*_length_of_line, flush=True)
        return

    
    @property
    def path(self) -> Path:     return self.__path
    @property
    def config(self) -> dict[str, Any]:   return self.__config
    
    @property
    def dtype(self) -> torch.dtype:     return self.__dtype
    @property
    def device(self) -> torch.device:   return self.__device
    
    @property
    def part_init(self) -> int:         return self.__part_init
    @property
    def num_repeat(self) -> int:        return self.__num_repeat
    @property
    def part_last(self) -> int:         return self.__part_last
    @property
    def batch_size(self) -> int:        return self.__batch_size
    
    @property
    def delta_t(self) -> float:         return self.__delta_t
    @property
    def max_t(self) -> float:           return self.__max_t
    @property
    def res_t(self) -> int:             return int(self.__max_t/self.__delta_t + 1.1)
    
    @property
    def dim(self) -> int:               return self.__dim
    @property
    def res_v(self) -> int:             return self.__res_v
    @property
    def max_v(self) -> float:           return self.__max_v
    @property
    def delta_v(self) -> float:         return (2*self.__max_v) / self.__res_v
    
    @property
    def v_perturb_rate(self) -> float:  return self.__v_perturb_rate
    @property
    def t_perturb_rate(self) -> float:  return self.__t_perturb_rate
    
    @property
    def exp_speed(self) -> list[float]: return self.__exp_speed
    @property
    def exp_angle(self) -> list[float]: return self.__exp_angle
    
    @property
    def storage(self) -> Path:          return self.__storage
    @property
    def log_name(self) -> Path:         return self.__log_name