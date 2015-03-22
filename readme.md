This repo contains the results of a course project I did for my machine
learning studies. My idea was to try out generating toy data samples
(20-bit vectors), and looking if the features that neural networks (RBM
and MLP) would find would match the latent variables that were used to
generate the data.

![Generate toy data -> Feed data to neural network -> evaluate if neurons found latent variables](report/method.svg "Diagram of the general idea")

## Reading the report

The project report is in the form of an IPython Notebook, allowing the
reader to rerun the experiments while reading, and to modify them to try
out new things.

To read and explore the report, clone the repo and run `ipython notebook report/report.ipynb`.

You will need:
* ipython,
* ipython-notebook
* the following python modules: (run `pip install <module>` to install modules)
  * numpy
  * matplotlib
  * sklearn
  * theano (only for MLP experiments)

Alternatively, you can take a read-only version of the report, as
[html](report/report.html) or [PDF](report/report.pdf).
