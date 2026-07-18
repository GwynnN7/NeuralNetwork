#include "cli.hpp"
#include "CLI/CLI.hpp"
#include "types.hpp"

Args parse_args(int argc, char *argv[]) {
    Args args;
    CLI::App app{"Neural Network Training"};

    std::map<std::string, DatasetType> dataset_map{
        {"xor", DatasetType::XOR}, {"xor_hot", DatasetType::XOR_HOT}, {"mnist", DatasetType::MNIST}};
    app.add_option("dataset", args.dataset_type, "Dataset type")
        ->transform(CLI::CheckedTransformer(dataset_map, CLI::ignore_case))
        ->required();
    app.add_option("-e,--epochs", args.epochs, "Number of epochs")->default_val(1000);
    app.add_option("-b,--batch", args.batch_size, "Batch size")->default_val(1);
    app.add_option("-n,--eta", args.eta, "Learning rate")->default_val(0.5);
    app.add_option("-l,--lambda", args.lambda, "Weight decay")->default_val(0);
    app.add_option("-a,--alpha", args.alpha, "Momentum")->default_val(0);

    std::map<std::string, TaskType> task_map{{"regression", TaskType::REGRESSION}, {"classification", TaskType::CLASSIFICATION}};
    app.add_option("-t,--task", args.task_type, "Task type")
        ->transform(CLI::CheckedTransformer(task_map, CLI::ignore_case))
        ->default_val(TaskType::REGRESSION);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        throw std::invalid_argument("Invalid command line arguments.");
    }

    std::cout << "\nTraining Configuration:" << "\n"
              << std::left << std::setw(25) << " • Epochs:" << args.epochs << "\n"
              << std::left << std::setw(25) << " • Batch Size:" << args.batch_size << "\n"
              << std::left << std::setw(25) << " • Learning Rate:" << args.eta << "\n"
              << std::left << std::setw(25) << " • Regularization:" << args.lambda << "\n"
              << std::left << std::setw(25) << " • Momentum:" << args.alpha << "\n\n";

    return args;
}