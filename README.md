# Parallel Coordinates Plot

Plot parallel coordinates using only matplotlib. Create multiple y-axes depending on your data, which can be linear, logarithmic or categorial. 

![Parallel Coordinates Plot](example/pcp_plot.svg)

You can also draw continues curves with colors matching the values of the last y-axis.

![Parallel Coordinates Plot with Cuves](example/pcp_plot_curves.svg)

## Example

A complete example with explanation and the plot above is given in the [jupyter notebook](example/example.ipynb).

## Conda Environment

1. Requirements:

   - Anaconda or Miniconda installation

2. Create the environment from the `environment.yml` file: 

   ```shell
   $ conda env create -f environment.yml
   ```

   - Activate the new environment with: `$ conda activate pcp`
   - Verify that the new environment was installed correctly: `$ conda env list`

3. Start a notebook server with:

   ```shell
   $ jupyter notebook
   ```

