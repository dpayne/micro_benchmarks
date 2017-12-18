#include <random>
#include <benchmark/benchmark.h>
#include "xxhash.h"
#include "t1ha.h"
#include "darbyhash.h"

namespace {
    static auto& chrs = "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
}

static std::string gen_random_str(int length)
{
    thread_local static std::mt19937 rg{std::random_device{}()};
    thread_local static std::uniform_int_distribution<std::string::size_type> pick(0, sizeof(chrs) - 2);

    std::string s;

    s.reserve(length);

    while(length--)
        s += chrs[pick(rg)];

    return s;
}

static void hash_bench_xxh64(benchmark::State &state)
{
    int count = 10000;
    std::vector<std::string> v;
    v.reserve( count );

    for (int n = 0; n < count; ++n)
    {
        v.push_back(gen_random_str(state.range(0)));
    }

    while (state.KeepRunning())
    {
        long hash = 0;
        for (std::string &str : v)
        {
            hash += XXH64( str.c_str(), str.size(), 0ul );
        }
        benchmark::DoNotOptimize(hash);
    }

    state.SetBytesProcessed(static_cast<long>(state.iterations()) *
                            static_cast<long>(count * state.range(0)));
    state.SetLabel(std::to_string(state.range(0)));
}

static void hash_bench_t1ha(benchmark::State &state)
{
    int count = 10000;
    std::vector<std::string> v;
    v.reserve( count );

    for (int n = 0; n < count; ++n)
    {
        v.push_back(gen_random_str(state.range(0)));
    }

    while (state.KeepRunning())
    {
        long hash = 0;
        for (std::string &str : v)
        {
            hash += t1ha( str.c_str(), str.size(), 0ul );
        }
        benchmark::DoNotOptimize(hash);
    }

    state.SetBytesProcessed(static_cast<long>(state.iterations()) *
                            static_cast<long>(count * state.range(0)));
    state.SetLabel(std::to_string(state.range(0)));
}

static void hash_bench_darbyhash(benchmark::State &state)
{
    int count = 10000;
    std::vector<std::string> v;
    v.reserve( count );

    for (int n = 0; n < count; ++n)
    {
        v.push_back(gen_random_str(state.range(0)));
    }

    while (state.KeepRunning())
    {
        long hash = 0;
        for (std::string &str : v)
        {
            hash += darbyhash( str.c_str(), str.size(), 0ul );
        }
        benchmark::DoNotOptimize(hash);
    }

    state.SetBytesProcessed(static_cast<long>(state.iterations()) *
                            static_cast<long>(count * state.range(0)));
    state.SetLabel(std::to_string(state.range(0)));
}

static void hash_bench_darbyhash_noavx(benchmark::State &state)
{
    int count = 10000;
    std::vector<std::string> v;
    v.reserve( count );

    for (int n = 0; n < count; ++n)
    {
        v.push_back(gen_random_str(state.range(0)));
    }

    while (state.KeepRunning())
    {
        long hash = 0;
        for (std::string &str : v)
        {
            hash += darbyhash_noavx( str.c_str(), str.size(), 0ul );
        }
        benchmark::DoNotOptimize(hash);
    }

    state.SetBytesProcessed(static_cast<long>(state.iterations()) *
                            static_cast<long>(count * state.range(0)));
    state.SetLabel(std::to_string(state.range(0)));
}

static void hash_bench_darbyhash_boring(benchmark::State &state)
{
    int count = 10000;
    std::vector<std::string> v;
    v.reserve( count );

    for (int n = 0; n < count; ++n)
    {
        v.push_back(gen_random_str(state.range(0)));
    }

    while (state.KeepRunning())
    {
        long hash = 0;
        for (std::string &str : v)
        {
            hash += darbyhash_boring( str.c_str(), str.size(), 0ul );
        }
        benchmark::DoNotOptimize(hash);
    }

    state.SetBytesProcessed(static_cast<long>(state.iterations()) *
                            static_cast<long>(count * state.range(0)));
    state.SetLabel(std::to_string(state.range(0)));
}

static void hash_bench_darbyhash_baseline(benchmark::State &state)
{
    int count = 10000;
    std::vector<std::string> v;
    v.reserve( count );

    for (int n = 0; n < count; ++n)
    {
        v.push_back(gen_random_str(state.range(0)));
    }

    while (state.KeepRunning())
    {
        long hash = 0;
        for (std::string &str : v)
        {
            hash += darbyhash_baseline( str.c_str(), str.size(), 0ul );
        }
        benchmark::DoNotOptimize(hash);
    }

    state.SetBytesProcessed(static_cast<long>(state.iterations()) *
                            static_cast<long>(count * state.range(0)));
    state.SetLabel(std::to_string(state.range(0)));
}

int32_t range_begin = 62;
int32_t range_end = 67;

BENCHMARK(hash_bench_xxh64)->Range(range_begin, range_end)->ReportAggregatesOnly(true);
//BENCHMARK(hash_bench_t1ha)->Range(range_begin, range_end)->ReportAggregatesOnly(true);
//BENCHMARK(hash_bench_darbyhash)->Range(range_begin, range_end)->ReportAggregatesOnly(true);
//BENCHMARK(hash_bench_darbyhash_noavx)->Range(range_begin, range_end)->ReportAggregatesOnly(true);
BENCHMARK(hash_bench_darbyhash_boring)->DenseRange(range_begin, range_end)->ReportAggregatesOnly(true);
BENCHMARK(hash_bench_darbyhash_baseline)->DenseRange(range_begin, range_end)->ReportAggregatesOnly(true);
