import time
import os
from database import RelationalDatabase
import matplotlib.pyplot as plt
import numpy as np

class Benchmarker:
    def __init__(self):
        self.Durations: dict[tuple[str, str], float] = {}
        self.TupleCounts: dict[tuple[str, str], tuple[int, int]] = {}

    def Benchmark2(self, database: RelationalDatabase, dataset_name: str, method: str):
        # Select the appropriate method function based on the method name
        if method.lower() == "alite":
            method_func = database.RunALITE
        elif method.lower() == "bicomnloj" and hasattr(database, "RunBIComNLoj"):
            method_func = database.RunBIComNLoj
        else:
            print(f"{method} is not a valid method or is not implemented.")
            return

        print("Test 1")
        # Measure initial tuple count
        input_tuples = database.TupleCount()

        # Benchmark the method execution time
        start_time = time.time()
        full_disjunction = method_func()
        end_time = time.time()
        
        print("Test 2")

        # Store the duration
        duration = end_time - start_time
        self.Durations[(dataset_name, method)] = duration

        # Attempt to get output tuple count
        try:
            output_tuples = full_disjunction.TupleCount()
        except AttributeError:
            output_tuples = -1

        # Store the input and output tuple counts
        self.TupleCounts[(dataset_name, method)] = (input_tuples, output_tuples)
        print(f"{method} took {duration:.2f} seconds on {dataset_name}, {input_tuples} -> {output_tuples}")

    def Benchmark(self, data_folder: str, dataset_name: str, method: str):
        db = RelationalDatabase()
        db.LoadFromFolder(data_folder)
        self.Benchmark2(db, dataset_name, method)

    def RunBenchmark(self, benchmark_folder: str):
        methods = ["ALITE", "BIComNLoj"]
        for root, dirs, _ in os.walk(benchmark_folder):
            if os.path.realpath(root) == os.path.realpath(benchmark_folder):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    db = RelationalDatabase()
                    db.LoadFromFolder(dir_path)
                    for method in methods:
                        self.Benchmark2(db, dir_name, "ALITE")

    def VisualizeDuration(self, max_datasets_visualized: int = 20):
        # Filter datasets and methods for visualization
        datasets = list(set([x[0] for x in self.Durations.keys()]))[:max_datasets_visualized]
        methods = sorted(list(set([x[1] for x in self.Durations.keys()])))

        # Sort datasets by ALITE duration for consistent ordering
        ALITE_durations = {ds: self.Durations[(ds, "ALITE")] for ds in datasets if ("ALITE" in methods)}
        sorted_datasets = sorted(datasets, key=lambda ds: ALITE_durations.get(ds, float('inf')))

        # Gather method durations
        method_durations = {method: [self.Durations[(ds, method)] for ds in sorted_datasets] for method in methods}

        # Plot configuration
        x = np.arange(len(sorted_datasets))  # Dataset label locations
        width = 0.25
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')
        for method, duration in method_durations.items():
            offset = width * multiplier
            ax.bar(x + offset, duration, width, label=method)
            multiplier += 1

        # Finalize plot details
        ax.set_ylabel('Time (s)')
        ax.set_title('Runtime of ALITE vs Other Methods')
        ax.set_xticks(x + width / 2, sorted_datasets)
        ax.tick_params(axis='x', labelrotation=90)
        ax.legend(loc='upper left', ncols=len(methods))
        ax.set_yscale('log')

    def VisualizeRuntimePerTuple(self, inputTuples: bool, reg_deg: int = 1):
        # Maximum tuple count
        max_count = max((self.TupleCounts[entry][0 if inputTuples else 1] for entry in self.TupleCounts), default=0)
        x_limit = max_count + 2000

        _, ax = plt.subplots()

        methods = sorted(list(set([x[1] for x in self.TupleCounts])))
        for method in methods:
            # Collect x and y values for scatter plot
            x_vals = [self.TupleCounts[entry][0 if inputTuples else 1] for entry in self.TupleCounts if entry[1] == method]
            y_vals = [self.Durations[entry] for entry in self.TupleCounts if entry[1] == method]

            # Plot data points and regression line
            ax.scatter(x_vals, y_vals, label=method)

            if reg_deg == 1:
                m, b = np.polyfit(x_vals, y_vals, deg=1)
                poly = lambda xx: m * xx + b
            elif reg_deg == 2:
                a, b, c = np.polyfit(x_vals, y_vals, deg=2)
                poly = lambda xx: a * xx**2 + b * xx + c

            x_span = np.linspace(0, x_limit)
            ax.plot(x_span, poly(x_span))

        # Set labels and scales
        tuple_type = 'Input' if inputTuples else 'Output'
        ax.set_xlabel(f'# of {tuple_type} Tuples')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'Runtime vs. {tuple_type} Tuple Count')
        ax.legend(loc='upper left')
        ax.set_xbound(lower=0, upper=x_limit)
        ax.set_yscale('log')
