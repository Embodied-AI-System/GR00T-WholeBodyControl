#include <iostream>
#include <chrono>
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <ctime>

// ONNX
#include <onnxruntime_cxx_api.h>

const int G1_NUM_MOTOR = 29;

class ONNXFreqTest {
  public:
    enum class DataGenerationMode {
      ZEROS, // All zeros
      RANDOM, // Small random values
      ONES // All ones
    };
  private:
    Ort::Env env;
    Ort::Session* policy_session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_node_names_str;
    std::vector<std::string> output_node_names_str;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<ONNXTensorElementDataType> input_types;
    std::vector<std::vector<float>> input_data_float;
    std::vector<std::vector<int64_t>> input_data_int64;
    DataGenerationMode data_mode = DataGenerationMode::RANDOM;
    std::string model_path;
  public:
    ONNXFreqTest(const std::string& model_file_path)
      : env(ORT_LOGGING_LEVEL_WARNING, "FreqTest"),
        model_path(model_file_path) {
      // Initialize random seed for fake data generation
      std::srand(static_cast<unsigned int>(std::time(nullptr)));

      try {
        std::cout << "Loading ONNX model: " << model_path << std::endl;
        Ort::SessionOptions policy_session_options;
        policy_session_options.SetIntraOpNumThreads(1);
        policy_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        policy_session = new Ort::Session(env, model_path.c_str(), policy_session_options);

        // Query input names from the ONNX model
        size_t num_input_nodes = policy_session->GetInputCount();
        std::cout << "Model has " << num_input_nodes << " inputs:" << std::endl;
        for (size_t i = 0; i < num_input_nodes; i++) {
          auto input_name = policy_session->GetInputNameAllocated(i, allocator);
          std::string name_str(input_name.get());

          // Get shape information
          auto input_type_info = policy_session->GetInputTypeInfo(i);
          auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
          auto shape = tensor_info.GetShape();
          auto element_type = tensor_info.GetElementType();

          std::cout << "  Input " << i << ": " << name_str << " - Shape: [";
          for (size_t j = 0; j < shape.size(); j++) {
            if (shape[j] == -1) {
              std::cout << "dynamic";
            } else {
              std::cout << shape[j];
            }
            if (j < shape.size() - 1) std::cout << ", ";
          }
          std::cout << "] - Type: " << element_type << std::endl;

          input_node_names_str.push_back(name_str);
          input_shapes.push_back(shape);
          input_types.push_back(element_type);
        }

        // Query output names from the ONNX model
        size_t num_output_nodes = policy_session->GetOutputCount();
        std::cout << "Model has " << num_output_nodes << " outputs:" << std::endl;
        for (size_t i = 0; i < num_output_nodes; i++) {
          auto output_name = policy_session->GetOutputNameAllocated(i, allocator);
          std::string name_str(output_name.get());

          // Get shape information
          auto output_type_info = policy_session->GetOutputTypeInfo(i);
          auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
          auto shape = tensor_info.GetShape();

          std::cout << "  Output " << i << ": " << name_str << " - Shape: [";
          for (size_t j = 0; j < shape.size(); j++) {
            if (shape[j] == -1) {
              std::cout << "dynamic";
            } else {
              std::cout << shape[j];
            }
            if (j < shape.size() - 1) std::cout << ", ";
          }
          std::cout << "] - Type: " << tensor_info.GetElementType() << std::endl;

          output_node_names_str.push_back(name_str);
        }

        // Initialize input data storage based on input shapes
        InitializeInputData();

        // Convert string vectors to const char* vectors for ONNX Runtime
        for (const auto& name : input_node_names_str) { input_node_names.push_back(name.c_str()); }
        for (const auto& name : output_node_names_str) { output_node_names.push_back(name.c_str()); }

        std::cout << "Successfully loaded policy model for frequency test" << std::endl;
      } catch (const Ort::Exception& e) { std::cerr << "Error loading policy model: " << e.what() << std::endl; }
    }

    ~ONNXFreqTest() { delete policy_session; }

    void InitializeInputData() {
      input_data_float.clear();
      input_data_int64.clear();
      input_data_float.resize(input_shapes.size());
      input_data_int64.resize(input_shapes.size());

      for (size_t i = 0; i < input_shapes.size(); i++) {
        // Calculate total size for this input
        int64_t total_size = 1;
        for (int64_t dim : input_shapes[i]) {
          if (dim > 0) { // Skip dynamic dimensions (-1)
            total_size *= dim;
          }
        }

        // Initialize storage based on data type
        if (input_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          input_data_float[i].resize(total_size, 0.0f);
        } else if (input_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          input_data_int64[i].resize(total_size, 0);
        }

        std::cout << "Initialized input " << i << " (" << input_node_names_str[i] << ") with " << total_size
                  << " elements" << std::endl;
      }
    }

    void CreateFakeInputData() {
      for (size_t i = 0; i < input_types.size(); i++) {
        if (input_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          for (size_t j = 0; j < input_data_float[i].size(); j++) {
            switch (data_mode) {
              case DataGenerationMode::ZEROS: input_data_float[i][j] = 0.0f; break;
              case DataGenerationMode::RANDOM:
                // Generate small random values between -0.1 and 0.1
                input_data_float[i][j] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
                break;
              case DataGenerationMode::ONES: input_data_float[i][j] = 1.0f; break;
            }
          }
        } else if (input_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          for (size_t j = 0; j < input_data_int64[i].size(); j++) {
            switch (data_mode) {
              case DataGenerationMode::ZEROS: input_data_int64[i][j] = 0; break;
              case DataGenerationMode::RANDOM:
                // Generate small random integers (0 to 9 for general use)
                input_data_int64[i][j] = rand() % 10;
                break;
              case DataGenerationMode::ONES: input_data_int64[i][j] = 1; break;
            }
          }
        }
      }
    }

    void RunInference() {
      // Update fake input data
      CreateFakeInputData();

      // Create input tensors dynamically based on model structure and data types
      std::vector<Ort::Value> input_tensors;

      for (size_t i = 0; i < input_shapes.size(); i++) {
        // Create tensor with appropriate data type
        if (input_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          Ort::Value tensor =
              Ort::Value::CreateTensor<float>(allocator.GetInfo(), input_data_float[i].data(), input_data_float[i].size(),
                                              input_shapes[i].data(), input_shapes[i].size());
          input_tensors.emplace_back(std::move(tensor));
        } else if (input_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          Ort::Value tensor =
              Ort::Value::CreateTensor<int64_t>(allocator.GetInfo(), input_data_int64[i].data(), input_data_int64[i].size(),
                                                input_shapes[i].data(), input_shapes[i].size());
          input_tensors.emplace_back(std::move(tensor));
        }
      }

      // Run inference
      auto output_tensors =
          policy_session->Run(Ort::RunOptions {nullptr}, input_node_names.data(), input_tensors.data(),
                              input_tensors.size(), output_node_names.data(), output_node_names.size());

      // For frequency testing, we don't need to process outputs
      // Just complete the inference to measure performance
    }

    void SetDataGenerationMode(DataGenerationMode mode) {
      data_mode = mode;
      std::string mode_name = (mode == DataGenerationMode::ZEROS)    ? "ZEROS"
                              : (mode == DataGenerationMode::RANDOM) ? "RANDOM"
                                                                     : "ONES";
      std::cout << "Set input data generation mode to: " << mode_name << std::endl;
    }

    static double Percentile(std::vector<double> values, double percentile) {
      if (values.empty()) { return 0.0; }
      std::sort(values.begin(), values.end());
      double index = (percentile / 100.0) * static_cast<double>(values.size() - 1);
      size_t lower = static_cast<size_t>(std::floor(index));
      size_t upper = static_cast<size_t>(std::ceil(index));
      if (lower == upper) { return values[lower]; }
      double weight = index - static_cast<double>(lower);
      return values[lower] * (1.0 - weight) + values[upper] * weight;
    }

    void RunFrequencyTest(int num_iterations = 1000, int warmup_iterations = 100) {
      std::cout << "\nRunning ONNX policy inference frequency test..." << std::endl;
      std::cout << "Warmup iterations: " << warmup_iterations << std::endl;
      std::cout << "Measured iterations: " << num_iterations << std::endl;

      std::string mode_name = (data_mode == DataGenerationMode::ZEROS)    ? "ZEROS"
                              : (data_mode == DataGenerationMode::RANDOM) ? "RANDOM"
                                                                          : "ONES";
      std::cout << "Using input data mode: " << mode_name << std::endl;

      for (int i = 0; i < warmup_iterations; ++i) { RunInference(); }

      std::vector<double> latencies_us;
      latencies_us.reserve(num_iterations);

      auto total_start = std::chrono::steady_clock::now();
      for (int i = 0; i < num_iterations; ++i) {
        auto iter_start = std::chrono::steady_clock::now();
        RunInference();
        auto iter_end = std::chrono::steady_clock::now();
        auto iter_duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(iter_end - iter_start);
        latencies_us.push_back(iter_duration.count());
      }
      auto total_end = std::chrono::steady_clock::now();

      auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
      double avg_time_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / latencies_us.size();
      double frequency_hz = 1000000.0 / avg_time_us;
      auto minmax = std::minmax_element(latencies_us.begin(), latencies_us.end());

      std::cout << "\n=== Results ===" << std::endl;
      std::cout << "Total measured time: " << total_duration.count() << " μs" << std::endl;
      std::cout << "Average time per inference: " << avg_time_us << " μs" << std::endl;
      std::cout << "p50 latency: " << Percentile(latencies_us, 50.0) << " μs" << std::endl;
      std::cout << "p90 latency: " << Percentile(latencies_us, 90.0) << " μs" << std::endl;
      std::cout << "p95 latency: " << Percentile(latencies_us, 95.0) << " μs" << std::endl;
      std::cout << "p99 latency: " << Percentile(latencies_us, 99.0) << " μs" << std::endl;
      std::cout << "Min latency: " << *minmax.first << " μs" << std::endl;
      std::cout << "Max latency: " << *minmax.second << " μs" << std::endl;
      std::cout << "Maximum frequency from average: " << frequency_hz << " Hz" << std::endl;
      std::cout << "Approx frequency from p95: " << (1000000.0 / Percentile(latencies_us, 95.0)) << " Hz" << std::endl;
      std::cout << "Approx frequency from p99: " << (1000000.0 / Percentile(latencies_us, 99.0)) << " Hz" << std::endl;
    }
};

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "ONNX Policy Frequency Test" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "Usage: " << argv[0] << " <model_file> [iterations] [data_mode]" << std::endl;
    std::cout << "  model_file: path to ONNX model file (required)" << std::endl;
    std::cout << "  iterations: number of inference iterations (default: 1000)" << std::endl;
    std::cout << "  data_mode: zeros|random|ones (default: random)" << std::endl;
    std::cout << "  warmup_iterations: number of warmup iterations (default: 100)" << std::endl;
    std::cout << "\nExample: " << argv[0] << " policy.onnx 5000 random 200" << std::endl;
    return 1;
  }

  std::string model_file = argv[1];
  int num_iterations = 1000;
  int warmup_iterations = 100;
  ONNXFreqTest::DataGenerationMode mode = ONNXFreqTest::DataGenerationMode::RANDOM;

  if (argc > 2) { num_iterations = std::atoi(argv[2]); }

  if (argc > 3) {
    std::string mode_str = argv[3];
    if (mode_str == "zeros") {
      mode = ONNXFreqTest::DataGenerationMode::ZEROS;
    } else if (mode_str == "ones") {
      mode = ONNXFreqTest::DataGenerationMode::ONES;
    } else if (mode_str == "random") {
      mode = ONNXFreqTest::DataGenerationMode::RANDOM;
    }
  }
  if (argc > 4) { warmup_iterations = std::atoi(argv[4]); }

  std::cout << "ONNX Policy Frequency Test" << std::endl;
  std::cout << "===========================" << std::endl;

  try {
    ONNXFreqTest freq_test(model_file);
    freq_test.SetDataGenerationMode(mode);
    freq_test.RunFrequencyTest(num_iterations, warmup_iterations);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
