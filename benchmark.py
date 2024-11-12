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