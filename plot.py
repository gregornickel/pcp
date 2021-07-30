import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from copy import deepcopy


def load_csv(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        data_list = list(reader)

    results = []
    for i in range(1, len(data_list)):
        result = []
        for j in data_list[i]:
            try:
                result.append(float(j))
            except ValueError:
                result.append(j)

        results.append(result)

    return results, data_list[0]


def check_data(data, labels):
    for i in range(len(data)):
        assert len(data[i]) is len(labels), "data dimension (%d) does not " \
            "match with labels (%d)" % (len(data[i]), len(labels))


def check_formatting(y_attribute, labels):
    if y_attribute:
        assert len(y_attribute) is len(labels), "dimension (%d) does not " \
            "match with labels (%d)" % (len(y_attribute), len(labels))
    else:
        y_attribute = [[]] * len(labels)
    return y_attribute


def set_y_type(y_type, data, colorbar):
    for i in range(len(y_type)):
        if not y_type[i]:
            if type(data[0][i]) is str:
                y_type[i] = "categorial"
            else:
                y_type[i] = "linear"
    if colorbar: 
        assert y_type[len(y_type) - 1] == "linear", "colorbar axis needs to " \
            "be linear"
    return y_type


# Automatically generate y_labels for string values
def set_y_labels(y_labels, data, y_type):
    for i in range(len(y_labels)):
        if y_type[i] == "categorial":
            y_label = deepcopy(y_labels[i])
            for j in range(len(data)):
                if data[j][i] not in y_label:
                    y_label.append(data[j][i])
            # Only sort if no labels specified
            if not len(y_labels[i]):
                y_label.sort()
            y_labels[i] = y_label
    return y_labels


def set_y_ticks(y_ticks, data, y_type, y_labels):
    for i in range(len(y_labels)):
        if y_type[i] == "categorial":
            y_ticks[i] = [len(y_labels[i])]
    return y_ticks


def replace_str_values(data, y_type, y_labels):
    for i in range(len(data[0])):
        if y_type[i] == "categorial":
            for j in range(len(data)):
                data[j][i] = y_labels[i].index(data[j][i])
    return np.array(data).transpose()


def set_y_lim(y_lim, data):
    for i in range(len(y_lim)):
        if not y_lim[i]:
            y_lim[i] = [np.min(data[i, :]), np.max(data[i, :])]
    return y_lim


def get_score(data, y_lim):
    y_min = y_lim[len(y_lim) - 1][0]
    y_max = y_lim[len(y_lim) - 1][1]
    score = (np.copy(data[len(y_lim) - 1, :]) - y_min) / (y_max - y_min)
    return score


# Rescale data of secondary y-axes to scale of first y-axis
def rescale_data(data, y_type, y_lim):
    scale = np.max(data[0, :])
    for i in range(1, len(y_lim)):
        y_min = y_lim[i][0]
        y_max = y_lim[i][1]
        if y_type[i] == "log":
            log_min = np.log10(y_min)
            log_max = np.log10(y_max)
            span = log_max - log_min
            data[i, :] = ((np.log10(data[i, :]) - log_min) / span) * scale
        else:
            data[i, :] = ((data[i, :] - y_min) / (y_max - y_min)) * scale
    return data


def get_path(data, i):
    n = data.shape[0] # number of y-axes
    verts = list(zip([x for x in np.linspace(0, n - 1, n * 3 - 2)], 
        np.repeat(data[:, i], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    return path


def pcp(data, 
             labels, 
             y_type=None, 
             y_labels=None, 
             y_ticks=None, 
             y_lim=None, 
             figsize=(10, 5), 
             rect=[0.125, 0.1, 0.75, 0.8], 
             curves=True,
             alpha=1.0,
             colorbar=True, 
             colorbar_width=0.03,
             cmap=plt.get_cmap("inferno")
             ):
    """
    Parallel Coordinates Plot 

    Parameters
    ----------
    data: nested array
        Inner arrays containing data for each curve.
    labels: list
        Labels for y-axes.
    y_type: list, optional
        Default "None" allows linear axes for numerical values and categorial 
        axes for data of type string. If y_type is passed, logarithmic axes are 
        also possible, e.g.  ["categorial", "linear", "log", ...].
    y_labels: nested array, optional
        Custom labels for ticks. 
    y_ticks: nested array, optional
        Custom number of ticks, should fit with the defined y_lim.
        Note: A wrong number of ticks for categorical axes is automatically 
        adjusted.
    y_lim: nested array, optional
        Custom min and max values for y-axes.
    figsize: (float, float), optional
        Width, height in inches.
    rect: array, optional
        [left, bottom, width, height], defines the position of the figure on
        the canvas. 
    curves: bool, optional
        If True, B-spline curve is drawn.
    alpha: float, optional
        Alpha value for blending the curves.
    colorbar: bool, optional
        If True, colorbar is drawn.
    colorbar_width: float, optional
        Defines the width of the colorbar.
    cmap: matplotlib.colors.Colormap, optional
        Specify colors for colorbar.
    
    Returns
    -------
    `~matplotlib.figure.Figure`
    """
    
    [left, bottom, width, height] = rect

    # Check data
    check_data(data, labels)
    y_type = check_formatting(y_type, labels)
    y_labels = check_formatting(y_labels, labels)
    y_ticks = check_formatting(y_ticks, labels)
    y_lim = check_formatting(y_lim, labels)

    # Setup data
    y_type = set_y_type(y_type, data, colorbar) 
    y_labels = set_y_labels(y_labels, data, y_type)
    y_ticks = set_y_ticks(y_ticks, data, y_type, y_labels)
    data = replace_str_values(data, y_type, y_labels)
    y_lim = set_y_lim(y_lim, data)
    score = get_score(data, y_lim)
    data = rescale_data(data, y_type, y_lim)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_axes([left, bottom, width, height])
    axes = [ax0] + [ax0.twinx() for i in range(data.shape[0] - 1)]

    # Plot curves
    for i in range(data.shape[1]):
        if colorbar:
            color = cmap(score[i])
        else:
            color = "blue"

        if curves:
            path = get_path(data, i)
            patch = PathPatch(path, facecolor="None", lw=1.5, alpha=alpha, 
                    edgecolor=color, clip_on=False)
            ax0.add_patch(patch)
        else:
            plt.plot(data[:, i], color=color, alpha=alpha, clip_on=False)

    # Format x-axis
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position("none")
    ax0.set_xlim([0, data.shape[0] - 1])
    ax0.set_xticks(range(data.shape[0]))
    ax0.set_xticklabels(labels)

    # Format y-axis
    for i, ax in enumerate(axes):
        ax.spines["left"].set_position(("axes", 1 / (len(labels) - 1) * i))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        
        ax.yaxis.set_ticks_position("left")
        ax.set_ylim(y_lim[i])
        if y_type[i] == "log":
            ax.set_yscale("log")
        if y_type[i] == "categorial":
            ax.set_yticks(range(y_ticks[i][0]))
        if y_type[i] == "linear" and y_ticks[i]:
            start = y_lim[i][0]
            stop = y_lim[i][1] + 1
            step = (stop - start) / y_ticks[i][0]
            ax.set_yticks(np.arange(start, stop, step))
        if y_labels[i]:
            ax.set_yticklabels(y_labels[i])
        
    if colorbar:
        bar = fig.add_axes([left + width, bottom, colorbar_width, height])
        norm = mpl.colors.Normalize(vmin=y_lim[i][0], vmax=y_lim[i][1])
        mpl.colorbar.ColorbarBase(bar, cmap=cmap, norm=norm, 
            orientation="vertical")
        bar.tick_params(size=0)
        bar.set_yticklabels([])

    return fig


if __name__ == "__main__":
    # Minimal working example
    results = [["ResNet", 0.0001, 4, 0.2],
               ["ResNet", 0.0003, 8, 1.0],
               ["DenseNet", 0.0005, 4, 0.65],
               ["DenseNet", 0.0007, 8, 0.45],
               ["DenseNet", 0.001, 2, 0.8]]
    labels = ["Network", "Learning rate", "Batchsize", "F-Score"]
    pcp(results, labels)
    plt.show()
