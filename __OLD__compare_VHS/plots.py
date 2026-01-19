from    typing              import  Callable, Sequence
from    pathlib             import  Path
import  numpy               as      np
import  torch
from    matplotlib.axes     import  Axes
from    matplotlib.figure   import  Figure
import  matplotlib.pyplot   as      plt

from    model_keys  import  *


def plot_train_history(
        train_histories:    dict[str, dict],
        save_dir:           str | Path,
        linewidth:          int = 1,
        dpi:                int = 600,
        window_size:        int = 1,
    ) -> None:
    axes: np.ndarray[Axes]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, dpi=dpi)

    # Configure the sliding window
    weight = torch.ones((1, 1, window_size), device="cpu") / window_size
    func_mean: Callable[[torch.Tensor], torch.Tensor] = \
        lambda x: torch.nn.functional.conv1d(
            x.reshape(1, 1, -1), weight
        ).flatten()
    
    # Plot the data loss
    axes[0, 0].set_title('Data loss (Train)')
    axes[1, 0].set_title('Data loss (Validation)')
    for k, v in train_histories.items():
        axes[0, 0].plot(
            func_mean(v['train_rel_error']),
            linewidth   = linewidth,
            label       = k+" (train)",
            **CONFIG__PLOT_OF_LOSS[k],
        )
        axes[1, 0].plot(
            func_mean(v['val_rel_error']),
            linewidth   = linewidth,
            label       = k+" (val)",
            **CONFIG__PLOT_OF_LOSS[k],
        )

    # Plot the conservation loss
    axes[0, 1].set_title('Conservation loss (Train)')
    axes[1, 1].set_title('Conservation loss (Validation)')
    for k, v in train_histories.items():
        axes[0, 1].plot(
            func_mean(v['train_rel_error_cons']),
            linewidth   = linewidth,
            label       = k+" (train)",
            **CONFIG__PLOT_OF_LOSS[k],
        )
        axes[1, 1].plot(
            func_mean(v['val_rel_error_cons']),
            linewidth   = linewidth,
            label       = k+" (val)",
            **CONFIG__PLOT_OF_LOSS[k],
        )

    for ax in axes.flatten():
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Relative $L^2$ error')
        ax.grid()
        ax.set_ylim(5e-5, 5e-2)
        ax.set_yscale('log')
        ax.legend()

    plt.tight_layout()
    filename = "train_histories.pdf" if window_size==1 else f"train_histories_slide{window_size}.pdf"
    plt.savefig(save_dir/filename, dpi=dpi)
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure
    
    return


def plot_rel_errors(
        arr_t:          torch.Tensor,
        rel_errors:     dict[str, torch.Tensor],
        model_keys:     Sequence[str],
        save_dir:       str | Path,
        linewidth:      int = 1,
        dpi:            int = 600,
        metric_order:   float = 2.0
    ) -> None:
    arr_t = arr_t.detach().cpu()
    fig = plt.figure(figsize=(10, 6), dpi=dpi)
    fig.suptitle(f"Average relative error ($p={metric_order}$)")
    for k in model_keys:
        _mean   = rel_errors[k].mean(dim=0)
        _std    = rel_errors[k].std(dim=0)
        _z = 1.0
        _lower  = torch.where(_mean-_z*_std>=0, _mean-_z*_std, 0)
        _upper  = _mean+_z*_std
        plt.plot(arr_t.cpu(), _mean, linewidth=linewidth, label=k, **CONFIG__PLOT_OF_LOSS[k])
        plt.fill_between(arr_t.cpu(), _lower, _upper, alpha=0.1, color=CONFIG__PLOT_OF_LOSS[k]['color'])
    plt.legend()
    plt.xlabel("$t$")
    plt.yscale('log')
    plt.ylim(5e-4, 1e-1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_dir/"test_rel_error.pdf", dpi=dpi)
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure
    
    return