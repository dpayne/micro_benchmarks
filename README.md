# micro_benchmarks
Various micro benchmarks

# Useful commands

Re-build and run a single benchmark
    rm -rf build; mkdir build; cd build; cmake ..; make VERBOSE=1; cd ..; ./build/MicroBenchmark --benchmark_filter=cachebench

Run a specific benchmark with a specific range
    ./build/MicroBenchmark --benchmark_filter=cache_bench/16

Run a specific benchmark with a specific range spending at least 2 seconds in the benchmark
    ./build/MicroBenchmark --benchmark_filter=cache_bench/16 --benchmark_min_time=2

Record a benchmark with perf stat
    perf stat ./build/MicroBenchmark --benchmark_filter=cache_bench/16 --benchmark_min_time=2
