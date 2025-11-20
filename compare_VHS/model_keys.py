# Keys of the models
K__FNO_PARAM    = "fno_param"
K__RFNO_PARAM   = "rfno_param"
K__PFNO_PARAM   = "pfno_param"
K__PRFNO_PARAM  = "prfno_param"
K__FNO_KERNEL   = "fno_kernel"
K__RFNO_KERNEL  = "rfno_kernel"
K__PFNO_KERNEL  = "pfno_kernel"
K__PRFNO_KERNEL = "prfno_kernel"


KEYS = tuple(
    [
        K__FNO_PARAM,
        K__RFNO_PARAM,
        K__PFNO_PARAM,
        K__PRFNO_PARAM,
        K__FNO_KERNEL,
        K__RFNO_KERNEL,
        K__PFNO_KERNEL,
        K__PRFNO_KERNEL,
    ]
)


CONFIG__PLOT_OF_LOSS: dict[str, dict[str, object]] = {
    K__FNO_PARAM:   {
        'color':        'black',
        'linestyle':    'dotted',
    },
    K__RFNO_PARAM:      {
        'color':        'red',
        'linestyle':    'dotted',
    },
    K__PFNO_PARAM:      {
        'color':        'blue',
        'linestyle':    'dotted',
    },
    K__PRFNO_PARAM:     {
        'color':        'orange',
        'linestyle':    'dotted',
    },
    K__FNO_KERNEL:      {
        'color':        'black',
        'linestyle':    'solid',
    },
    K__RFNO_KERNEL:     {
        'color':        'red',
        'linestyle':    'solid',
    },
    K__PFNO_KERNEL:     {
        'color':        'blue',
        'linestyle':    'solid',
    },
    K__PRFNO_KERNEL:    {
        'color':        'orange',
        'linestyle':    'solid',
    },
}