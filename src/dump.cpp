#include "dump.hpp"

#include <fstream>

// Dump and Load everythin needed to reconstruct the model

void dump(const std::string& model_path, const Args& args, Network* network) {
    std::ofstream dump_file(model_path + "/model.bin", std::ios::binary);
    if (!dump_file.is_open()) {
        std::println(stderr, "Failed to open {} for writing.", model_path + "/model.bin");
        return;
    } else {
        std::println("\n- Dumping model to: {}", model_path + "/model.bin");
    }

    int magic_number = 0x4E4E4554; // "NNET" in hex
    dump_file.write(reinterpret_cast<const char*>(&magic_number), sizeof(magic_number));

    dump_file.write(reinterpret_cast<const char*>(&args.hidden_activation), sizeof(args.hidden_activation));
    dump_file.write(reinterpret_cast<const char*>(&args.output_activation), sizeof(args.output_activation));

    int num_layers = static_cast<int>(args.net_struct.size());
    dump_file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    dump_file.write(reinterpret_cast<const char*>(args.net_struct.data()), num_layers * sizeof(int));

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

Network* load_model(const std::string& model_path, Args* args) {
    std::ifstream dump_file(model_path + "/model.bin", std::ios::binary);
    if (!dump_file.is_open()) {
        std::println(stderr, "Failed to open {} for reading.", model_path + "/model.bin");
        return nullptr;
    } else {
        std::println("\n- Loading model from: {}", model_path + "/model.bin");
    }

    int magic_number;
    dump_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    if (magic_number != 0x4E4E4554) { // "NNET" in hex
        std::println(stderr, "Invalid model file format");
        return nullptr;
    }

    dump_file.read(reinterpret_cast<char*>(&args->hidden_activation), sizeof(args->hidden_activation));
    dump_file.read(reinterpret_cast<char*>(&args->output_activation), sizeof(args->output_activation));

    int num_layers;
    dump_file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    args->net_struct.resize(num_layers);
    dump_file.read(reinterpret_cast<char*>(args->net_struct.data()), num_layers * sizeof(int));

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

    return new Network(*args, layers_weights, layers_biases);
}