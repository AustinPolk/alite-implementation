# Getting Started

All code should be ran through the *test_suite* Jupyter Notebook. All benchmarking has been automated such that it can be controlled from this notebook, which can be executed in a compatible IDE such as Visual Studio Code. All visualizations contained in the final report were also generated in this notebook, and available for confirmation.

## Environment Setup (done through Visual Studio Code on a Windows machine)
1. Ensure that the Jupyter and Python extensions are installed in Visual Studio Code
2. Configure your Jupyter notebook to use Python 3.11
3. Run the pip install code snippet in the *test_suite* file to install all of the necessary modules.
    a. Ensure the modules install successfully, if they do not then look at the error code to resolve.

## Running Benchmarking Code
1. Open test_suite.ipynb
2. Run the first code cell to install packages if not done so already
3. Run the second cell to initialize the benchmarking object
4. Run the benchmarks to confirm results from the final report
    a. To run all Align and Real benchmarks, run the third and fourth cells (should take roughly 30 minutes)
    b. To run selected datasets from these benchmarks, run the fifth and sixth cells
    c. All intermediate data from running an integration benchmark on a dataset can be found in the TestData folder under the associated dataset name
5. Generate visualizations and generate statistics by running all cells below the previously mentioned cells
