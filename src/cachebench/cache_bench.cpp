#include <atomic>
#include <benchmark/benchmark.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <list>
#include <deque>

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

static void push_pop_list(benchmark::State &state)
{
    int count = state.range(0);
    std::list<std::string> v;
    v.reserve( count );

    std::vector<std::string> backup;
    backup.reserve( count );

    for (int n = 0; n < count; ++n)
    {
        v.push_back(gen_random_str(100000));
        backup.push_back(gen_random_str(100000));
    }

    while (state.KeepRunning())
    {
        for ( auto i = 0 ; i < 100000; ++i )
        {
            v.pop_front();
            v.push_back[backup[i]];
        }
        benchmark::DoNotOptimize(v);
    }
}
static void push_pop_deque(benchmark::State &state)
{
    int count = state.range(0);
    std::deque<std::string> v;
    v.reserve( count );

    std::vector<std::string> backup;
    backup.reserve( count );

    for (int n = 0; n < count; ++n)
    {
        v.push_back(gen_random_str(100000));
        backup.push_back(gen_random_str(100000));
    }

    while (state.KeepRunning())
    {
        for ( auto i = 0 ; i < 100000; ++i )
        {
            v.pop_front();
            v.push_back[backup[i]];
        }
        benchmark::DoNotOptimize(v);
    }
}

BENCHMARK(push_pop_list)->DenseRange(10, 12)->ReportAggregatesOnly(true);
BENCHMARK(push_pop_deque)->DenseRange(10, 12)->ReportAggregatesOnly(true);
