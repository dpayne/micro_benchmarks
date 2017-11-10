#include <atomic>
#include <benchmark/benchmark.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>

static void cache_bench(benchmark::State &state)
{
    int bytes = 1 << state.range(0);
    int count = (bytes / sizeof(int)) / 2;
    std::vector<int> v;
    std::random_device
        rd; // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> int_dis(std::numeric_limits<int>::min(),
                                            std::numeric_limits<int>::max());

    for (int n = 0; n < count; ++n)
    {
        v.push_back(int_dis(gen));
    }

    std::uniform_int_distribution<> indicies_dis(0, v.size() - 1);
    std::vector<int> indicies;
    for (int n = 0; n < v.size(); ++n)
    {
        indicies.push_back(indicies_dis(gen));
    }

    while (state.KeepRunning())
    {
        long sum = 0;
        for (int &i : indicies)
        {
            sum += v[i];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetBytesProcessed(static_cast<long>(state.iterations()) *
                            static_cast<long>(bytes));
    state.SetLabel(std::to_string(bytes / 1024) + "kb");
}

BENCHMARK(cache_bench)->DenseRange(10, 28)->ReportAggregatesOnly(true);
