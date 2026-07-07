#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "control_policy.hpp"
#include "encoder.hpp"

namespace {

struct LatencySample {
  int iteration = 0;
  double start_us = 0.0;
  double end_us = 0.0;
  long long wall_start_us = 0;
  long long wall_end_us = 0;
  double latency_us = 0.0;
};

double Percentile(std::vector<double> values, double percentile) {
  if (values.empty()) { return 0.0; }
  std::sort(values.begin(), values.end());
  double index = (percentile / 100.0) * static_cast<double>(values.size() - 1);
  size_t lower = static_cast<size_t>(std::floor(index));
  size_t upper = static_cast<size_t>(std::ceil(index));
  if (lower == upper) { return values[lower]; }
  double weight = index - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

void FillInput(TPinnedVector<float>& input) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  for (auto& value : input) { value = dist(rng); }
}

void PrintStats(const std::string& label, const std::vector<double>& latencies_us) {
  if (latencies_us.empty()) {
    std::cerr << "No latency samples collected." << std::endl;
    return;
  }

  double mean = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / latencies_us.size();
  auto minmax = std::minmax_element(latencies_us.begin(), latencies_us.end());

  std::cout << "\n=== " << label << " Results ===" << std::endl;
  std::cout << "samples: " << latencies_us.size() << std::endl;
  std::cout << "mean: " << mean << " us" << std::endl;
  std::cout << "p50: " << Percentile(latencies_us, 50.0) << " us" << std::endl;
  std::cout << "p90: " << Percentile(latencies_us, 90.0) << " us" << std::endl;
  std::cout << "p95: " << Percentile(latencies_us, 95.0) << " us" << std::endl;
  std::cout << "p99: " << Percentile(latencies_us, 99.0) << " us" << std::endl;
  std::cout << "min: " << *minmax.first << " us" << std::endl;
  std::cout << "max: " << *minmax.second << " us" << std::endl;
  std::cout << "freq from mean: " << (1000000.0 / mean) << " Hz" << std::endl;
  std::cout << "freq from p95: " << (1000000.0 / Percentile(latencies_us, 95.0)) << " Hz" << std::endl;
  std::cout << "freq from p99: " << (1000000.0 / Percentile(latencies_us, 99.0)) << " Hz" << std::endl;
}

std::vector<double> ExtractLatencies(const std::vector<LatencySample>& samples) {
  std::vector<double> latencies_us;
  latencies_us.reserve(samples.size());
  for (const auto& sample : samples) { latencies_us.push_back(sample.latency_us); }
  return latencies_us;
}

void WriteSamplesCsv(const std::string& path, const std::vector<LatencySample>& samples) {
  if (path.empty()) { return; }

  std::ofstream out(path);
  if (!out) { throw std::runtime_error("failed to open CSV output path: " + path); }

  out << "iteration,start_us,end_us,wall_start_us,wall_end_us,latency_us\n";
  for (const auto& sample : samples) {
    out << sample.iteration << "," << sample.start_us << "," << sample.end_us << "," << sample.wall_start_us << ","
        << sample.wall_end_us << "," << sample.latency_us << "\n";
  }
  std::cout << "wrote latency samples: " << path << std::endl;
}

template <typename RunFn>
std::vector<LatencySample> Benchmark(
    RunFn&& run, int warmup_iterations, int measured_iterations, double period_ms = 0.0) {
  for (int i = 0; i < warmup_iterations; ++i) {
    if (!run()) { throw std::runtime_error("warmup inference failed"); }
  }

  std::vector<LatencySample> samples;
  samples.reserve(measured_iterations);
  auto benchmark_start = std::chrono::steady_clock::now();
  const auto period = std::chrono::duration<double, std::milli>(period_ms);
  for (int i = 0; i < measured_iterations; ++i) {
    if (period_ms > 0.0) {
      auto scheduled = benchmark_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(period * i);
      std::this_thread::sleep_until(scheduled);
    }
    auto start = std::chrono::steady_clock::now();
    auto wall_start = std::chrono::system_clock::now();
    if (!run()) { throw std::runtime_error("measured inference failed"); }
    auto wall_end = std::chrono::system_clock::now();
    auto end = std::chrono::steady_clock::now();
    samples.push_back({
        i,
        std::chrono::duration<double, std::micro>(start - benchmark_start).count(),
        std::chrono::duration<double, std::micro>(end - benchmark_start).count(),
        std::chrono::duration_cast<std::chrono::microseconds>(wall_start.time_since_epoch()).count(),
        std::chrono::duration_cast<std::chrono::microseconds>(wall_end.time_since_epoch()).count(),
        std::chrono::duration<double, std::micro>(end - start).count(),
    });
  }
  return samples;
}

void PrintUsage(const char* argv0) {
  std::cout << "TensorRT SONIC Frequency Test" << std::endl;
  std::cout << "Usage: " << argv0
            << " <encoder|policy> <model.onnx> [iterations=5000] [warmup=200] [fp32|fp16=fp32] [graph|no_graph=graph] [csv_path] [period_ms=0]"
            << std::endl;
  std::cout << "       " << argv0
            << " round <encoder.onnx> <decoder.onnx> [iterations=5000] [warmup=200] [fp32|fp16=fp32] [graph|no_graph=graph] [csv_path] [period_ms=0]"
            << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  " << argv0 << " encoder policy/low_latency/model_encoder.onnx 5000 200 fp32 graph" << std::endl;
  std::cout << "  " << argv0 << " policy  policy/low_latency/model_decoder.onnx 5000 200 fp32 graph" << std::endl;
  std::cout << "  " << argv0 << " round   policy/low_latency/model_encoder.onnx policy/low_latency/model_decoder.onnx 5000 200 fp16 graph" << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  std::string engine_type = argv[1];

  if (engine_type == "round") {
    if (argc < 4) {
      PrintUsage(argv[0]);
      return 1;
    }

    std::string encoder_model_path = argv[2];
    std::string decoder_model_path = argv[3];
    int measured_iterations = argc > 4 ? std::atoi(argv[4]) : 5000;
    int warmup_iterations = argc > 5 ? std::atoi(argv[5]) : 200;
    bool use_fp16 = argc > 6 ? std::string(argv[6]) == "fp16" : false;
    bool use_graph = argc > 7 ? std::string(argv[7]) != "no_graph" : true;
    std::string csv_path = argc > 8 ? argv[8] : "";
    double period_ms = argc > 9 ? std::atof(argv[9]) : 0.0;

    std::cout << "Encoder model: " << encoder_model_path << std::endl;
    std::cout << "Policy model: " << decoder_model_path << std::endl;
    std::cout << "Engine type: round" << std::endl;
    std::cout << "Precision: " << (use_fp16 ? "fp16" : "fp32") << std::endl;
    std::cout << "CUDA graph: " << (use_graph ? "enabled" : "disabled") << std::endl;
    std::cout << "Warmup iterations: " << warmup_iterations << std::endl;
    std::cout << "Measured iterations: " << measured_iterations << std::endl;
    if (!csv_path.empty()) { std::cout << "CSV output: " << csv_path << std::endl; }
    if (period_ms > 0.0) { std::cout << "Scheduled period: " << period_ms << " ms" << std::endl; }

    try {
      EncoderEngine encoder;
      PolicyEngine policy;

      if (!encoder.Initialize(encoder_model_path, use_fp16)) { return 2; }
      if (!policy.Initialize(decoder_model_path, use_fp16)) { return 2; }

      FillInput(encoder.GetInputBuffer());
      FillInput(policy.GetInputBuffer());

      if (!encoder.Encode()) { return 3; }
      if (!policy.Infer()) { return 3; }

      if (use_graph && !encoder.CaptureGraph()) { return 4; }
      if (use_graph && !policy.CaptureGraph()) { return 4; }

      auto latencies = Benchmark(
          [&encoder, &policy]() { return encoder.Encode() && policy.Infer(); },
          warmup_iterations, measured_iterations, period_ms);
      PrintStats("round", ExtractLatencies(latencies));
      WriteSamplesCsv(csv_path, latencies);
      return 0;
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return -1;
    }
  }

  std::string model_path = argv[2];
  int measured_iterations = argc > 3 ? std::atoi(argv[3]) : 5000;
  int warmup_iterations = argc > 4 ? std::atoi(argv[4]) : 200;
  bool use_fp16 = argc > 5 ? std::string(argv[5]) == "fp16" : false;
  bool use_graph = argc > 6 ? std::string(argv[6]) != "no_graph" : true;
  std::string csv_path = argc > 7 ? argv[7] : "";
  double period_ms = argc > 8 ? std::atof(argv[8]) : 0.0;

  std::cout << "Model: " << model_path << std::endl;
  std::cout << "Engine type: " << engine_type << std::endl;
  std::cout << "Precision: " << (use_fp16 ? "fp16" : "fp32") << std::endl;
  std::cout << "CUDA graph: " << (use_graph ? "enabled" : "disabled") << std::endl;
  std::cout << "Warmup iterations: " << warmup_iterations << std::endl;
  std::cout << "Measured iterations: " << measured_iterations << std::endl;
  if (!csv_path.empty()) { std::cout << "CSV output: " << csv_path << std::endl; }
  if (period_ms > 0.0) { std::cout << "Scheduled period: " << period_ms << " ms" << std::endl; }

  try {
    if (engine_type == "policy") {
      PolicyEngine engine;
      if (!engine.Initialize(model_path, use_fp16)) { return 2; }
      FillInput(engine.GetInputBuffer());
      if (!engine.Infer()) { return 3; }
      if (use_graph && !engine.CaptureGraph()) { return 4; }
      auto latencies = Benchmark([&engine]() { return engine.Infer(); }, warmup_iterations, measured_iterations, period_ms);
      PrintStats("policy", ExtractLatencies(latencies));
      WriteSamplesCsv(csv_path, latencies);
      return 0;
    }

    if (engine_type == "encoder") {
      EncoderEngine engine;
      if (!engine.Initialize(model_path, use_fp16)) { return 2; }
      FillInput(engine.GetInputBuffer());
      if (!engine.Encode()) { return 3; }
      if (use_graph && !engine.CaptureGraph()) { return 4; }
      auto latencies = Benchmark([&engine]() { return engine.Encode(); }, warmup_iterations, measured_iterations, period_ms);
      PrintStats("encoder", ExtractLatencies(latencies));
      WriteSamplesCsv(csv_path, latencies);
      return 0;
    }

    std::cerr << "Unknown engine type: " << engine_type << std::endl;
    PrintUsage(argv[0]);
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
}
