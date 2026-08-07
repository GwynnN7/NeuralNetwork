#include "model.hpp"

#include "network.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Load a grid search of models from a CSV file
std::vector<Model> Model::load_grid_search(const std::string& filename) {
    std::vector<Model> grid_search;

    // Helper lambda function to convert a string to lowercase for case-insensitive mapping
    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        return s;
    };

    // Read the CSV file and parse each row into a Model object
    for (const CsvRow& csv_row : load_csv(filename, ',', true)) {
        const std::vector<std::string>& row = csv_row.cells;
        const int row_number = csv_row.line;

        // Validate that the row has the expected number of columns. The 12th (beta1) column is optional
        if (row.size() != HYPERPARAMS_NUM && row.size() != HYPERPARAMS_NUM_OPT) {
            throw std::runtime_error("Grid row " + std::to_string(row_number) + ": invalid number of columns");
        }

        Model model;
        model.id = parse_cell<int>(row[0], filename, row_number, "id");

        // Parse the network structure from the second column (e.g., "64-32-10" for a network with 3 hidden layers).
        std::string num;
        std::stringstream struct_ss(row[1]);
        while (std::getline(struct_ss, num, '-')) {
            const int neurons = parse_cell<int>(trim(num), filename, row_number, "layer size");
            if (neurons <= 0) {
                throw std::runtime_error("Grid row " + std::to_string(row_number) + ": layer sizes must be positive");
            }
            model.net_struct.push_back(neurons);
        }
        if (model.net_struct.empty()) {
            throw std::runtime_error("Grid row " + std::to_string(row_number) + ": the 'net' column must have at least one hidden layer");
        }

        // Parse the remaining hyperparameters from the row
        try {
            model.hidden_activation = Maps::str_to_activation.at(to_lower(row[2]));
            model.output_activation = Maps::str_to_activation.at(to_lower(row[3]));
            model.init_type = Maps::str_to_init.at(to_lower(row[4]));
            model.opt_type = Maps::str_to_optimizer.at(to_lower(row[5]));
            model.loss_type = Maps::str_to_loss.at(to_lower(row[6]));
        } catch (const std::out_of_range&) {
            throw std::runtime_error("Grid row " + std::to_string(row_number) + ": unknown activation, init, optimizer or loss name");
        }

        model.batch_size = parse_cell<int>(row[7], filename, row_number, "batch");
        model.eta = parse_cell<Scalar>(row[8], filename, row_number, "eta");
        model.lambda = parse_cell<Scalar>(row[9], filename, row_number, "lambda");
        model.alpha = parse_cell<Scalar>(row[10], filename, row_number, "alpha");
        model.beta1 = (row.size() == HYPERPARAMS_NUM_OPT) ? parse_cell<Scalar>(row[11], filename, row_number, "beta1") : ADAM_B1;

        // Avoid out-of-range hyperparameters
        if (model.batch_size < 0) {
            throw std::runtime_error("Grid row " + std::to_string(row_number) + ": batch size must be non-negative (0 means full batch)");
        }
        if (!(model.eta > 0)) {
            throw std::runtime_error("Grid row " + std::to_string(row_number) + ": eta must be positive");
        }
        if (model.lambda < 0) {
            throw std::runtime_error("Grid row " + std::to_string(row_number) + ": lambda must be non-negative");
        }
        if (model.alpha < 0 || model.alpha >= 1) {
            throw std::runtime_error("Grid row " + std::to_string(row_number) + ": alpha must be in [0, 1)");
        }
        if (model.beta1 < 0 || model.beta1 >= 1) {
            throw std::runtime_error("Grid row " + std::to_string(row_number) + ": beta1 must be in [0, 1)");
        }

        grid_search.push_back(model);
    }

    if (grid_search.empty()) {
        throw std::runtime_error("Grid search file does not contain any valid models: " + filename);
    }

    // Check that all model ids are unique
    std::vector<int> ids;
    ids.reserve(grid_search.size());
    for (const auto& model : grid_search) {
        ids.push_back(model.id);
    }
    std::ranges::sort(ids);
    if (std::ranges::adjacent_find(ids) != ids.end()) {
        throw std::runtime_error("Grid search file contains models with duplicate ids: " + filename);
    }

    return grid_search;
}

void Model::print() const {
    std::println("\nTraining Configuration:");

    std::println(" • {:<25}{}", "Batch Size:", batch_size);
    std::println(" • {:<25}{}", "Learning Rate:", eta);
    std::println(" • {:<25}{}", "Regularization:", lambda);
    std::println(" • {:<25}{}", "Momentum:", alpha);
    if (opt_type == OptimizerType::ADAM) {
        std::println(" • {:<25}{}", "Adam Beta1:", beta1);
    }
    std::println(" • {:<25}{}", "Hidden Activation:", Maps::activation_to_str.at(hidden_activation));
    std::println(" • {:<25}{}", "Output Activation:", Maps::activation_to_str.at(output_activation));
    std::println(" • {:<25}{}", "Weight Init:", Maps::init_to_str.at(init_type));
    std::println(" • {:<25}{}", "Optimizer:", Maps::optimizer_to_str.at(opt_type));
    std::println(" • {:<25}{}", "Loss Function:", Maps::loss_to_str.at(loss_type));
    std::println(" • {:<25}{}", "Task Type:", Maps::task_to_str.at(task));
}

// Dump and Load everything needed to reconstruct the model
namespace Serializer {
namespace {
constexpr std::uint32_t MAGIC = 0x4E4E4554; // "NNET" in hex

// Write a value of type T to the output stream in binary format
template <typename T>
void write_data(std::ostream& out, const T& value) {
    static_assert(std::is_trivial_v<T>, "write_data can only be used with Plain Old Data types");

    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Read a value of type T from the input stream in binary format
template <typename T>
T read_data(std::istream& in, const char* action) {
    static_assert(std::is_trivial_v<T>, "read_data can only be used with Plain Old Data types");

    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in || in.gcount() != static_cast<std::streamsize>(sizeof(T))) {
        throw std::runtime_error(std::string("Error reading ") + action);
    }
    return value;
}

// Read a fixed number of bytes from the input stream into the destination buffer
void read_array(std::istream& in, void* dest, std::size_t bytes, const char* action) {
    static_assert(std::is_trivial_v<void*>, "read_array can only be used with Plain Old Data types");

    in.read(reinterpret_cast<char*>(dest), static_cast<std::streamsize>(bytes));
    if (!in || in.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error(std::string("Error reading ") + action);
    }
}

// Read a dimension (rows or columns) from the input stream
std::int32_t read_dimension(std::istream& in, const char* action) {
    const std::int32_t value = read_data<std::int32_t>(in, action);
    if (value <= 0) {
        throw std::runtime_error("Invalid model file: " + std::string(action) + " must be positive");
    }
    return value;
}
} // namespace

// Dump the model and its weights/biases to a binary file
void dump_model(const std::filesystem::path& file, const Model& model, const Network& network) {
    std::ofstream dump_file(file, std::ios::binary);
    if (!dump_file.is_open()) {
        throw std::runtime_error("Failed to open " + file.string() + " for writing.");
    }
    std::print("\n- Dumping model to: {}", file.string());

    write_data(dump_file, MAGIC);
    write_data(dump_file, static_cast<std::uint32_t>(sizeof(Scalar)));

    // Enums are serialized as their integer values
    write_data(dump_file, static_cast<std::int32_t>(model.hidden_activation));
    write_data(dump_file, static_cast<std::int32_t>(model.output_activation));
    write_data(dump_file, static_cast<std::int32_t>(model.opt_type));
    write_data(dump_file, static_cast<std::int32_t>(model.loss_type));
    write_data(dump_file, static_cast<std::int32_t>(model.task));

    const std::vector<const DenseLayer*> dense_layers = network.getDenseLayers();

    write_data(dump_file, static_cast<std::int32_t>(model.net_struct.size()));
    for (int neurons : model.net_struct) {
        write_data(dump_file, static_cast<std::int32_t>(neurons));
    }
    // Explicit layer count, checked on load
    write_data(dump_file, static_cast<std::int32_t>(dense_layers.size()));

    for (const auto* layer : dense_layers) {
        const Matrix& weights = layer->getWeights();
        const Vector& biases = layer->getBiases();

        write_data(dump_file, static_cast<std::int32_t>(weights.rows()));
        write_data(dump_file, static_cast<std::int32_t>(weights.cols()));
        dump_file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(Scalar));

        write_data(dump_file, static_cast<std::int32_t>(biases.size()));
        dump_file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(Scalar));
    }

    dump_file.flush();
    if (!dump_file) {
        throw std::runtime_error("Failed while writing model file: " + file.string());
    }
}

// Load a model and its weights/biases from a binary file
std::unique_ptr<Network> load_model(const std::filesystem::path& file, const Dataset& dataset) {
    std::ifstream dump_file(file, std::ios::binary);
    if (!dump_file.is_open()) {
        std::println(stderr, "Failed to open {} for reading", file.string());
        return nullptr;
    }
    std::println("\n- Loading model from: {}", file.string());

    try {
        if (read_data<std::uint32_t>(dump_file, "magic number") != MAGIC) {
            std::println(stderr, "Invalid model file format");
            return nullptr;
        }
        const std::uint32_t scalar_size = read_data<std::uint32_t>(dump_file, "scalar size");
        if (scalar_size != sizeof(Scalar)) {
            std::println(stderr, "Model file was written with {} byte scalars, this build uses {} byte", scalar_size, sizeof(Scalar));
            return nullptr;
        }

        // Helper lambda to read an enum value
        auto read_enum = [&dump_file](const char* action, int max_value) {
            const std::int32_t value = read_data<std::int32_t>(dump_file, action);
            if (value < 0 || value > max_value) {
                throw std::runtime_error(std::string("Invalid model reading ") + action);
            }
            return value;
        };

        Model model;
        model.hidden_activation = static_cast<ActivationType>(read_enum("hidden activation", static_cast<int>(ActivationType::SOFTMAX)));
        model.output_activation = static_cast<ActivationType>(read_enum("output activation", static_cast<int>(ActivationType::SOFTMAX)));
        model.opt_type = static_cast<OptimizerType>(read_enum("optimizer", static_cast<int>(OptimizerType::ADAM)));
        model.loss_type = static_cast<LossType>(read_enum("loss", static_cast<int>(LossType::CCE)));
        model.task = static_cast<TaskType>(read_enum("task", static_cast<int>(TaskType::CLASSIFICATION)));

        if (model.task != dataset.task) {
            std::println(stderr, "Model was trained for a different task type than the provided dataset");
            return nullptr;
        }

        const std::int32_t num_layers = read_data<std::int32_t>(dump_file, "layer count");
        if (num_layers <= 0) {
            throw std::runtime_error("Model file declares an implausible layer count: " + std::to_string(num_layers));
        }
        model.net_struct.reserve(static_cast<size_t>(num_layers));
        for (std::int32_t i = 0; i < num_layers; ++i) {
            model.net_struct.push_back(read_dimension(dump_file, "layer size"));
        }

        const std::int32_t num_dense = read_data<std::int32_t>(dump_file, "dense layer count");
        if (num_dense != num_layers + 1) {
            throw std::runtime_error("Mismatch between the number of dense layers in the model file and the expected number of layers");
        }

        // Read the weights and biases for each layer
        std::vector<Matrix> layers_weights;
        std::vector<Vector> layers_biases;
        layers_weights.reserve(static_cast<size_t>(num_dense));
        layers_biases.reserve(static_cast<size_t>(num_dense));

        for (std::int32_t i = 0; i < num_dense; ++i) {
            const std::int32_t rows = read_dimension(dump_file, "weight row count");
            const std::int32_t cols = read_dimension(dump_file, "weight column count");

            Matrix weights(rows, cols);
            read_array(dump_file, weights.data(), static_cast<std::size_t>(rows) * cols * sizeof(Scalar), "weights");

            const std::int32_t bias_size = read_dimension(dump_file, "bias count");
            if (bias_size != rows) {
                throw std::runtime_error("Model file layer has " + std::to_string(rows) + " outputs but " + std::to_string(bias_size) + " biases");
            }

            Vector biases(bias_size);
            read_array(dump_file, biases.data(), static_cast<std::size_t>(bias_size) * sizeof(Scalar), "biases");

            layers_weights.push_back(std::move(weights));
            layers_biases.push_back(std::move(biases));
        }

        // Check that the model's input/output dimensions match the dataset
        if (static_cast<int>(layers_weights.front().cols()) != dataset.num_features) {
            throw std::runtime_error("Model input size does not match dataset feature count");
        }
        if (static_cast<int>(layers_weights.back().rows()) != dataset.num_classes) {
            throw std::runtime_error("Model output size does not match dataset class count");
        }

        return std::make_unique<Network>(model, layers_weights, layers_biases, false);
    } catch (const std::exception& e) {
        std::println(stderr, "Failed to load {}: {}", file.string(), e.what());
        return nullptr;
    }
}
} // namespace Serializer