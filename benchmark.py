import time
import os
from database import RelationalDatabase

class Benchmarker:
    def __init__(self):
        self.Durations: dict[tuple[str, str], float] = {}
        self.TupleCounts: dict[tuple[str, str], tuple[int, int]] = {}

    def Benchmark(self, data_folder: str, dataset_name: str, method: str):
        
        db = RelationalDatabase()
        db.LoadFromFolder(data_folder)

        if method.lower() == "alite":
            method_func = db.RunALITE
        elif method.lower() == "bicomnloj":
            method_func = db.RunBIComNLoj
        else:
            print(f"{method} is not a valid method.")

        input_tuples = db.TupleCount()

        start_time = time.time()
        full_disjunction = method_func()
        end_time = time.time()

        duration = end_time - start_time
        self.Durations[(dataset_name, method)] = duration

        output_tuples = full_disjunction.TupleCount()
        self.TupleCounts[(dataset_name, method)] = (input_tuples, output_tuples)

        print(f"{method} took {duration} seconds on {dataset_name}, {input_tuples} -> {output_tuples}")

    def RunBenchmark(self, benchmark_folder: str):
        methods = ["ALITE", "BIComNLoj"]
        for root, dirs, files in os.walk(benchmark_folder):
            if os.path.realpath(root) == os.path.realpath(benchmark_folder):
                for dir in dirs:
                    dirpath = os.path.join(root, dir)
                    for method in methods:
                        self.Benchmark(dirpath, dir, method)