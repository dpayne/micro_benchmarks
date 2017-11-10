#include <atomic>
#include <benchmark/benchmark.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>

static void clamp_bench(benchmark::State &state)
{
    int count = state.range(0);

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

    while (state.KeepRunning())
    {
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
        for (int & i : v)
        {
            i = i > 255 ? 255 : i;
        }
        benchmark::DoNotOptimize(v);
    }

    state.SetItemsProcessed(static_cast<long>(state.iterations()) *
                            static_cast<long>(v.size()));
}

BENCHMARK(clamp_bench)->Range(1 << 10, 1 << 20)->ReportAggregatesOnly(true);
