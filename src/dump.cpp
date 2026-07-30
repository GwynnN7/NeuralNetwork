#include "dump.hpp"

#include <fstream>

// Dump and Load everything needed to reconstruct the model

void dump_model(const std::string& file, const Model& model, Network* network) {
    std::ofstream dump_file(file, std::ios::binary);
    if (!dump_file.is_open()) {
        std::println(stderr, "Failed to open {} for writing.", file);
        return;
    } else {
        std::println("\n- Dumping model to: {}", file);
    }

    int magic_number = 0x4E4E4554; // "NNET" in hex
    dump_file.write(reinterpret_cast<const char*>(&magic_number), sizeof(magic_number));

    dump_file.write(reinterpret_cast<const char*>(&model.hidden_activation), sizeof(model.hidden_activation));
    dump_file.write(reinterpret_cast<const char*>(&model.output_activation), sizeof(model.output_activation));

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

Network* load_model(const std::string& file) {
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

    int num_layers;
    dump_file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    model.net_struct.resize(num_layers);
    dump_file.read(reinterpret_cast<char*>(model.net_struct.data()), num_layers * sizeof(int));

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

    return new Network(model, layers_weights, layers_biases);
}