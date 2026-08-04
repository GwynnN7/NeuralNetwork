#include "model.hpp"

#include "network.hpp"
#include "types.hpp"

#include <fstream>
#include <map>
#include <print>
#include <string>
#include <vector>

// Load a grid search of models from a CSV file
std::vector<Model> Model::load_grid_search(const std::string& filename) {
    std::vector<Model> grid_search;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open grid search file: " + filename);
    }

    // Helper lambda function to convert a string to lowercase for case-insensitive mapping
    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        return s;
    };

    std::string line;
    bool is_first_line = true;

    while (std::getline(file, line)) {
        // Skip empty lines or lines that contain only whitespace
        if (line.empty() || line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }

        // Skip the header line if it contains non-numeric characters
        if (is_first_line) {
            is_first_line = false;
            if (!std::isdigit(line[0]))
                continue;
        }

        // Load the row of the CSV into a vector of strings, splitted by commas
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        // Validate that the row has the expected number of columns
        if (row.size() != HYPERPARAMS_NUM) {
            throw std::runtime_error("Invalid row in CSV. Expected " + std::to_string(HYPERPARAMS_NUM) + " columns, found " + std::to_string(row.size()));
        }

        Model model;
        model.id = std::stoi(row[0]);

        // Parse the network structure from the second column (e.g., "64-32-10" for a network with 3 hidden layers)
        std::string num;
        std::stringstream struct_ss(row[1]);
        while (std::getline(struct_ss, num, '-')) {
            if (!num.empty())
                model.net_struct.push_back(std::stoi(num));
        }

        // Parse the remaining hyperparameters from the row
        try {
            model.hidden_activation = Maps::str_to_activation.at(to_lower(row[2]));
            model.output_activation = Maps::str_to_activation.at(to_lower(row[3]));
            model.init_type = Maps::str_to_init.at(to_lower(row[4]));
            model.opt_type = Maps::str_to_optimizer.at(to_lower(row[5]));
            model.loss_type = Maps::str_to_loss.at(to_lower(row[6]));
        } catch (const std::out_of_range&) {
            throw std::runtime_error("Invalid activation or init type in CSV row " + std::to_string(model.id));
        }

        model.batch_size = std::stoi(row[7]);
        model.eta = static_cast<Scalar>(std::stod(row[8]));
        model.lambda = static_cast<Scalar>(std::stod(row[9]));
        model.alpha = static_cast<Scalar>(std::stod(row[10]));
        grid_search.push_back(model);
    }

    file.close();
    return grid_search;
}

void Model::print() const {
    std::println("\nTraining Configuration:");

    std::println(" • {:<25}{}", "Batch Size:", batch_size);
    std::println(" • {:<25}{}", "Learning Rate:", eta);
    std::println(" • {:<25}{}", "Regularization:", lambda);
    std::println(" • {:<25}{}", "Momentum:", alpha);
    std::println(" • {:<25}{}", "Hidden Activation:", Maps::activation_to_str.at(hidden_activation));
    std::println(" • {:<25}{}", "Output Activation:", Maps::activation_to_str.at(output_activation));
    std::println(" • {:<25}{}", "Weight Init:", Maps::init_to_str.at(init_type));
    std::println(" • {:<25}{}", "Optimizer:", Maps::optimizer_to_str.at(opt_type));
    std::println(" • {:<25}{}", "Loss Function:", Maps::loss_to_str.at(loss_type));
    std::println(" • {:<25}{}", "Task Type:", Maps::task_to_str.at(task));
}

// Dump and Load everything needed to reconstruct the model
namespace Serializer {
void dump_model(const std::string& file, const Model& model, Network* network) {
    std::ofstream dump_file(file, std::ios::binary);
    if (!dump_file.is_open()) {
        std::println(stderr, "Failed to open {} for writing.", file);
        return;
    } else {
        std::print("\n- Dumping model to: {}", file);
    }

    int magic_number = 0x4E4E4554; // "NNET" in hex
    dump_file.write(reinterpret_cast<const char*>(&magic_number), sizeof(magic_number));

    dump_file.write(reinterpret_cast<const char*>(&model.hidden_activation), sizeof(model.hidden_activation));
    dump_file.write(reinterpret_cast<const char*>(&model.output_activation), sizeof(model.output_activation));
    dump_file.write(reinterpret_cast<const char*>(&model.opt_type), sizeof(model.opt_type));
    dump_file.write(reinterpret_cast<const char*>(&model.loss_type), sizeof(model.loss_type));
    dump_file.write(reinterpret_cast<const char*>(&model.task), sizeof(model.task));

    int num_layers = static_cast<int>(model.net_struct.size());
    dump_file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    dump_file.write(reinterpret_cast<const char*>(model.net_struct.data()), num_layers * sizeof(int));

    std::vector<const DenseLayer*> dense_layers = network->getDenseLayers();
    for (const auto* layer : dense_layers) {
        int rows = static_cast<int>(layer->getWeights().rows());
        int cols = static_cast<int>(layer->getWeights().cols());
        dump_file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        dump_file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        dump_file.write(reinterpret_cast<const char*>(layer->getWeights().data()), rows * cols * sizeof(Scalar));

        int bias_size = static_cast<int>(layer->getBiases().size());
        dump_file.write(reinterpret_cast<const char*>(&bias_size), sizeof(bias_size));
        dump_file.write(reinterpret_cast<const char*>(layer->getBiases().data()), bias_size * sizeof(Scalar));
    }
}

std::unique_ptr<Network> load_model(const std::string& file, const Dataset& dataset) {
    std::ifstream dump_file(file, std::ios::binary);
    if (!dump_file.is_open()) {
        std::println(stderr, "Failed to open {} for reading.", file);
        return nullptr;
    } else {
        std::println("\n- Loading model from: {}", file);
    }

    int magic_number;
    dump_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    if (magic_number != 0x4E4E4554) { // "NNET" in hex
        std::println(stderr, "Invalid model file format");
        return nullptr;
    }
    Model model;
    dump_file.read(reinterpret_cast<char*>(&model.hidden_activation), sizeof(model.hidden_activation));
    dump_file.read(reinterpret_cast<char*>(&model.output_activation), sizeof(model.output_activation));
    dump_file.read(reinterpret_cast<char*>(&model.opt_type), sizeof(model.opt_type));
    dump_file.read(reinterpret_cast<char*>(&model.loss_type), sizeof(model.loss_type));
    dump_file.read(reinterpret_cast<char*>(&model.task), sizeof(model.task));
    if (model.task != dataset.task) {
        std::println(stderr, "Model was trained for a different task type than the provided dataset.");
        return nullptr;
    }

    int num_layers;
    dump_file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    model.net_struct.resize(num_layers);
    dump_file.read(reinterpret_cast<char*>(model.net_struct.data()), num_layers * sizeof(int));

    // Read the weights and biases for each layer
    std::vector<Matrix> layers_weights;
    std::vector<Vector> layers_biases;
    for (int i = 0; i < num_layers + 1; ++i) {
        int rows, cols;
        dump_file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        dump_file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        Matrix weights(rows, cols);
        dump_file.read(reinterpret_cast<char*>(weights.data()), rows * cols * sizeof(Scalar));

        int bias_size;
        dump_file.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));

        Vector biases(bias_size);
        dump_file.read(reinterpret_cast<char*>(biases.data()), bias_size * sizeof(Scalar));

        layers_weights.push_back(weights);
        layers_biases.push_back(biases);
    }

    // Create a new Network instance using the loaded model and weights/biases
    return std::make_unique<Network>(model, layers_weights, layers_biases);
}
} // namespace Serializer