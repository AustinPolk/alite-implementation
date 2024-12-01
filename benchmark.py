import time
import os
from database import RelationalDatabase
import matplotlib.pyplot as plt
import numpy as np

class Benchmarker:
    def __init__(self):
        self.Durations: dict[tuple[str, str], float] = {}
        self.TupleCounts: dict[tuple[str, str], tuple[int, int]] = {}
        self.ClusterDurations: dict[str, float] = {}
        self.ClusterQuality: dict[str, list[float]] = {}
        self.ClusterParameters: dict[str, list[int]] = {}
        self.SilhouetteScores: dict[str, dict[int, float]] = {}

        if not os.path.exists('TestData'):
            os.mkdir('TestData')

    def Benchmark2(self, database: RelationalDatabase, dataset_name: str, method: str):
        # Select the appropriate method function based on the method name
        if method.lower() == "alite":
            method_func = database.RunALITE
        else:
            print(f"{method} is not a valid method or is not implemented.")
            return
        
        # create data folder in TestData
        dataset_output_folder = f'TestData\\{dataset_name}'
        if not os.path.exists(dataset_output_folder):
            os.mkdir(dataset_output_folder)

        # Measure initial tuple count
        input_tuples = database.TupleCount()

        # Benchmark the method execution time
        start_time = time.time()
        full_disjunction = method_func(dataset_output_folder)
        end_time = time.time()

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

    def RunBenchmarks(self, align_benchmark_folder: str ,integration_benchmark_folder: str):
        # run clustering quality benchmarks for the align benchmark folder
        if align_benchmark_folder:
            for root, dirs, _ in os.walk(align_benchmark_folder):
                if os.path.realpath(root) == os.path.realpath(align_benchmark_folder):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        db = RelationalDatabase()
                        db.LoadFromFolder(dir_path)
                        self.ClusteringQualityStatistics(db, dir_name)
        
        # run the integration benchmarks for the integration benchmark folder
        if integration_benchmark_folder:
            methods = ["ALITE"]
            for root, dirs, _ in os.walk(integration_benchmark_folder):
                if os.path.realpath(root) == os.path.realpath(integration_benchmark_folder):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        db = RelationalDatabase()
                        db.LoadFromFolder(dir_path)
                        for method in methods:
                            self.Benchmark2(db, dir_name, method)

    def VisualizeDuration(self, max_datasets_visualized: int = 20, log_scale: bool = False):
        # Filter datasets and methods for visualization
        datasets = list(set([x[0] for x in self.Durations.keys()]))[:max_datasets_visualized]
        methods = sorted(list(set([x[1] for x in self.Durations.keys()])))

        # Sort datasets by ALITE duration for consistent ordering
        ALITE_durations = {ds: self.Durations[(ds, "ALITE")] for ds in datasets}
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
        ax.set_title('ALITE Runtime')
        reduce_width = lambda x: f'{x[:7]}...' if len(x) > 7 else x 
        ax.set_xticks(x + width / 2, [reduce_width(x) for x in sorted_datasets])
        ax.tick_params(axis='x', labelrotation=90)
        #ax.legend(loc='upper left', ncols=len(methods))
        if log_scale:
            ax.set_yscale('log')

    def VisualizeRuntimePerTuple(self, inputTuples: bool, reg_deg: int = 1, log_scale: bool = False):
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
        #ax.legend(loc='upper left')
        ax.set_xbound(lower=0, upper=x_limit)
        if log_scale:
            ax.set_yscale('log')

    def ClusteringQualityStatistics(self, database: RelationalDatabase, dataset_name: str):
        if not database.IntegrationIDsAssigned:
            start = time.time()
            database.AssignIntegrationIDs()
            end = time.time()
            self.ClusterDurations[dataset_name] = end - start
        self.SilhouetteScores[dataset_name] = database.SilhouetteScores
        
        # in this case, a "Negative" is a relation between a column from one table and a column from
        # another table that does not exist. A "Positive" is a relation between two such columns that
        # does exist. In this benchmark, it can be assumed that columns with the same name have a Positive
        # relation, while those with different names have a Negative relation
        true_negatives = 0
        false_negatives = 0
        true_positives = 0
        false_positives = 0
        
        unique_columns = set()
        table_count = len(database.Tables)

        # take each pair of tables (including a table and itself), and take each pair of columns from 
        # these tables to compare their names and assigned integration IDs
        for i in range(table_count):
            for j in range(i, table_count):
                table = database.Tables[i]
                other_table = database.Tables[j]
                for columnID, columnName in table.ColumnNames.items():
                    for other_columnID, other_columnName in other_table.ColumnNames.items():
                        unique_columns.add(columnName)
                        unique_columns.add(other_columnName)
                        if columnName == other_columnName:
                            # this is a positive relation
                            if columnID == other_columnID:
                                # is positive, flagged as positive
                                true_positives += 1
                            else:
                                # is positive, flagged as negative
                                false_negatives += 1
                        else:
                            # this is a negative relation
                            if columnID == other_columnID:
                                # is negative, flagged as positive
                                false_positives += 1
                            else:
                                # is negative, flagged as negative
                                true_negatives += 1

        precision = (true_positives) / (true_positives + false_positives)
        recall = (true_positives) / (true_positives + false_negatives)
        accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
        f1 = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)

        print(f"For dataset {dataset_name}:")
        print(f"True positives: {true_positives}")
        print(f"False positives: {false_positives}")
        print(f"True negatives: {true_negatives}")
        print(f"False negatives: {false_negatives}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")
        print(f"F1 score: {f1}")

        all_stats = [true_positives, false_positives, true_negatives, false_negatives, precision, recall, accuracy, f1]
        self.ClusterQuality[dataset_name] = all_stats

        # also record parameters that may affect the cluster quality, in particular
        # the number of input tables, the minimum and maximum number of total columns in the FD, 
        # the predicted number of columns, and the actual number of unique columns
        all_params = [len(database.Tables)]
        all_params.extend(database.ColumnClusterSizes)
        all_params.append(len(unique_columns))
        self.ClusterParameters[dataset_name] = all_params

    def VisualizeClusterStatistics(self, x_param: str, y_param: str, scatter: bool, with_reg: bool):
        x = []
        y = []
        x_label = None
        y_label = None
        for dataset in self.ClusterQuality:
            true_positives, false_positives, true_negatives, false_negatives, precision, recall, accuracy, f1 = self.ClusterQuality[dataset]
            table_count, min_columns, max_columns, predicted_columns, actual_columns = self.ClusterParameters[dataset]
            duration = self.ClusterDurations[dataset]

            if x_param == 'tp':
                x.append(true_positives)
                if not x_label:
                    x_label = "True Positives"
            elif x_param == 'fp':
                x.append(false_positives)
                if not x_label:
                    x_label = "False Positives"
            elif x_param == 'tn':
                x.append(true_negatives)
                if not x_label:
                    x_label = "True Negatives"
            elif x_param == 'fn':
                x.append(false_negatives)
                if not x_label:
                    x_label = "False Negatives"
            elif x_param == 'p':
                x.append(precision)
                if not x_label:
                    x_label = "Precision"
            elif x_param == 'r':
                x.append(recall)
                if not x_label:
                    x_label = "Recall"
            elif x_param == 'a':
                x.append(accuracy)
                if not x_label:
                    x_label = "Accuracy"
            elif x_param == 'f':
                x.append(f1)
                if not x_label:
                    x_label = "F1 Score"
            elif x_param == 'tc':
                x.append(table_count)
                if not x_label:
                    x_label = "Input Table Count"
            elif x_param == 'min':
                x.append(min_columns)
                if not x_label:
                    x_label = "Minimum Column Count"
            elif x_param == 'max':
                x.append(max_columns)
                if not x_label:
                    x_label = "Maximum Column Count"
            elif x_param == 'pre':
                x.append(predicted_columns)
                if not x_label:
                    x_label = "Predicted Column Count"
            elif x_param == 'act':
                x.append(actual_columns)
                if not x_label:
                    x_label = "Actual Column Count"
            elif x_param == 'dur':
                x.append(duration)
                if not x_label:
                    x_label = "Clustering Runtime"

            if y_param == 'tp':
                y.append(true_positives)
                if not y_label:
                    y_label = "True Positives"
            elif y_param == 'fp':
                y.append(false_positives)
                if not y_label:
                    y_label = "False Positives"
            elif y_param == 'tn':
                y.append(true_negatives)
                if not y_label:
                    y_label = "True Negatives"
            elif y_param == 'fn':
                y.append(false_negatives)
                if not y_label:
                    y_label = "False Negatives"
            elif y_param == 'p':
                y.append(precision)
                if not y_label:
                    y_label = "Precision"
            elif y_param == 'r':
                y.append(recall)
                if not y_label:
                    y_label = "Recall"
            elif y_param == 'a':
                y.append(accuracy)
                if not y_label:
                    y_label = "Accuracy"
            elif y_param == 'f':
                y.append(f1)
                if not y_label:
                    y_label = "F1 Score"
            elif y_param == 'tc':
                y.append(table_count)
                if not y_label:
                    y_label = "Input Table Count"
            elif y_param == 'min':
                y.append(min_columns)
                if not y_label:
                    y_label = "Minimum Column Count"
            elif y_param == 'max':
                y.append(max_columns)
                if not y_label:
                    y_label = "Maximum Column Count"
            elif y_param == 'pre':
                y.append(predicted_columns)
                if not y_label:
                    y_label = "Predicted Column Count"
            elif y_param == 'act':
                y.append(actual_columns)
                if not y_label:
                    y_label = "Actual Column Count"
            elif y_param == 'dur':
                y.append(duration)
                if not y_label:
                    y_label = "Clustering Runtime"

        # sort the x and y coords so that a line or scatter plot can be used
        xy = zip(x, y)
        sorted_xy = sorted(xy, key = lambda k: k[0])
        sorted_x, sorted_y = zip(*sorted_xy)

        # now plot either scatter or line plot
        if scatter:
            plt.scatter(sorted_x, sorted_y)
        else:
            plt.plot(sorted_x, sorted_y)

        # optionally include a best-fit line
        if with_reg:
            m, b = np.polyfit(x, y, deg=1)
            poly = lambda xx: m * xx + b
            
            x_span = np.linspace(min(x) * 0.9, max(x) * 1.1)
            plt.plot(x_span, poly(x_span))

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"Column Alignment: {x_label} vs. {y_label}")
        plt.show()

    def VisualizeSilhouetteScores(self, dataset_name: str):
        x = []
        y = []
        maximum_x = -1
        maximum_y = -1
        minimum_y = 1
        for n_clusters, score in self.SilhouetteScores[dataset_name].items():
            x.append(n_clusters)
            y.append(score)
            if score > maximum_y:
                maximum_y = score
                maximum_x = n_clusters
            if score < minimum_y:
                minimum_y = score

        plt.plot(x, y)
        plt.vlines(x = maximum_x, ymin=minimum_y, ymax=min(1, maximum_x + 0.1), colors='blue')
        plt.xlabel("# of Column Clusters")
        plt.ylabel("Silhouette Score")
        plt.title(f"Column Clustering for {dataset_name[:10]}...")
        plt.show()

