from database import RelationalDatabase
from table import RelationalTable
from benchmark import Benchmarker
import os

bm = Benchmarker()
db = RelationalDatabase()
# db.LoadFromFolder("Benchmark/Real Benchmark/school_report")
# bm.Benchmark2(db, "Benchmark/Real Benchmark/school_report", "ALITE")
# db.LoadFromFolder("Benchmark/Minimum/500spend")
# bm.Benchmark2(db, "Benchmark/Minimum/500spend", "ALITE")
db.LoadFromFolder("Benchmark/Real Benchmark/cihr")
bm.Benchmark2(db, "Benchmark/Real Benchmark/cihr", "ALITE")