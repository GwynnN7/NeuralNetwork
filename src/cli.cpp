#include "cli.hpp"

#include "CLI/CLI.hpp"
#include "types.hpp"

Args parse_args(int argc, char* argv[]) {
    Args args;
    CLI::App app{"Neural Network Training"};

    std::map<std::string, DatasetType> dataset_map{{"xor", DatasetType::XOR}, {"xor_hot", DatasetType::XOR_HOT}, {"mnist", DatasetType::MNIST}};
    app.add_option("dataset", args.dataset_type, "Dataset type")
        ->transform(CLI::CheckedTransformer(dataset_map, CLI::ignore_case))
        ->required();
    std::map<std::string, ActivationType> activation_map{{"sigmoid", ActivationType::SIGMOID}, {"relu", ActivationType::RELU}, {"tanh", ActivationType::TANH}, {"softmax", ActivationType::SOFTMAX}, {"linear", ActivationType::LINEAR}};
    app.add_option("--hidden", args.hidden_activation, "Activation function for hidden layers")
        ->transform(CLI::CheckedTransformer(activation_map, CLI::ignore_case))
        ->default_val(ActivationType::SIGMOID);
    app.add_option("--output", args.output_activation, "Activation function for output layer")
        ->transform(CLI::CheckedTransformer(activation_map, CLI::ignore_case))
        ->default_val(ActivationType::LINEAR);
    app.add_option("--init", args.init_type, "Weight initialization method")
        ->transform(CLI::CheckedTransformer(std::map<std::string, InitializationType>{{"random", InitializationType::RANDOM}, {"lecun", InitializationType::LECUN}, {"glorot", InitializationType::GLOROT}, {"he", InitializationType::HE}}, CLI::ignore_case))
        ->default_val(InitializationType::LECUN);

    app.add_option("--network", args.net_struct, "Network structure")->default_val(std::vector<int>{2, 1});
    app.add_option("--epochs", args.epochs, "Number of epochs")->default_val(1000);
    app.add_option("--batch_size", args.batch_size, "Batch size")->default_val(0);
    app.add_option("--eta", args.eta, "Learning rate")->default_val(0.5);
    app.add_option("--lambda", args.lambda, "Weight decay")->default_val(0);
    app.add_option("--alpha", args.alpha, "Momentum")->default_val(0);

    app.add_option("--train_ratio", args.train_ratio, "Training set ratio")->default_val(0.8);
    app.add_option("--dataset_ratio", args.dataset_ratio, "Subset of dataset used (when applicable)")->default_val(1.0);
    app.add_option("--log", args.log_file, "Output file for loss log")->default_val("log.csv");
    app.add_option("--dump", args.dump_file, "Dump file for model weights");
    app.add_option("--load", args.load_file, "Load model weights from file");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        throw std::invalid_argument("Invalid command line arguments.");
    }

    return args;
}