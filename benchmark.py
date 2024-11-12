import time
import os
from database import RelationalDatabase

class Benchmarker:
    def __init__(self):
        self.Durations: dict[tuple[str, str], float] = {}
        self.TupleCounts: dict[tuple[str, str], tuple[int, int]] = {}

    def Benchmark2(self, database: RelationalDatabase, dataset_name: str, method: str):
        if method.lower() == "alite":
            method_func = database.RunALITE
        elif method.lower() == "bicomnloj":
            method_func = database.RunBIComNLoj
        else:
            method_func = print(f"{method} is not a valid method.")

        input_tuples = database.TupleCount()

        start_time = time.time()
        full_disjunction = method_func()
        end_time = time.time()

        duration = end_time - start_time
        self.Durations[(dataset_name, method)] = duration

        try:
            output_tuples = full_disjunction.TupleCount()
        except:
            output_tuples = -1
        
        self.TupleCounts[(dataset_name, method)] = (input_tuples, output_tuples)

        print(f"{method} took {duration} seconds on {dataset_name}, {input_tuples} -> {output_tuples}")

    def Benchmark(self, data_folder: str, dataset_name: str, method: str):
        db = RelationalDatabase()
        db.LoadFromFolder(data_folder)
        self.Benchmark2(db, dataset_name, method)

    def RunBenchmark(self, benchmark_folder: str):
        methods = ["ALITE", "BIComNLoj"]
        for root, dirs, files in os.walk(benchmark_folder):
            if os.path.realpath(root) == os.path.realpath(benchmark_folder):
                for dir in dirs:
                    dirpath = os.path.join(root, dir)
                    db = RelationalDatabase()
                    db.LoadFromFolder(dirpath)
                    for method in methods:
                        self.Benchmark2(db, dir, method)

    def VisualizeDuration(self, max_datasets_visualized: int = 20):
        import matplotlib.pyplot as plt
        import numpy as np

        durs = self.Durations

        datasets = list(set([x[0] for x in durs.keys()]))[:max_datasets_visualized]
        methods = list(set([x[1] for x in durs.keys()]))

        t_datasets = tuple(datasets)

        ALITE_durations = {}
        for entry in durs:
            entry_dataset, entry_method = entry
            if entry_method == "ALITE" and entry_dataset in datasets:
                ALITE_durations[entry_dataset] = durs[entry]

        sorted_datasets = sorted(datasets, key = lambda x: ALITE_durations[x])

        method_durations = {}
        for method in methods:
            d = []
            for dataset in sorted_datasets:
                dur = durs[(dataset, method)]
                d.append(dur)
            method_durations[method] = tuple(d)

        x = np.arange(len(t_datasets))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for method, duration in method_durations.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, duration, width, label=method)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Time (s)')
        ax.set_title('Runtime of ALITE vs Other methods')
        ax.set_xticks([])
        ax.tick_params(axis='x', labelrotation=90)
        ax.legend(loc='upper left', ncols=len(methods))

    def VisualizeRuntimePerTuple(self, inputTuples: bool, reg_deg: int = 1):
        import matplotlib.pyplot as plt
        import numpy as np
        
        max_count = 0

        tupleCounts = {}
        runTimes = {}
        for entry in self.TupleCounts:
            _, entry_method = entry
            if entry_method not in tupleCounts:
                tupleCounts[entry_method] = []
            if entry_method not in runTimes:
                runTimes[entry_method] = []
            
            tuples = self.TupleCounts[entry][0 if inputTuples else 1]
            duration = self.Durations[entry]
            tupleCounts[entry_method].append(tuples)
            runTimes[entry_method].append(duration)

            if tuples > max_count:
                max_count = tuples

        _, ax = plt.subplots()

        x_limit = max_count + 2000

        for method in tupleCounts:
            x = tupleCounts[method]
            y = runTimes[method]

            ax.scatter(x, y, label = method)

            if reg_deg == 1:
                m, b = np.polyfit(x, y, deg=1)
                poly = lambda xx: m * xx + b
            elif reg_deg == 2:
                a, b, c = np.polyfit(x, y, deg=2)
                poly = lambda xx: a * xx * xx + b * xx + c
            xsp = np.linspace(start=0, stop=x_limit)
            ax.plot(xsp, poly(xsp))

        io = 'Input' if inputTuples else 'Output'
        ax.set_ylabel('Time (s)')
        ax.set_xlabel(f'# of {io} tuples')
        ax.set_title(f'Runtime vs. {io} tuple count')
        ax.legend(loc='upper left')
        ax.set_xbound(lower = 0, upper=x_limit)
        ax.set_yscale('log')

