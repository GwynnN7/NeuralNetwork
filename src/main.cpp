#include "cli.hpp"
#include "dataset.hpp"
#include "model.hpp"
#include "network.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <csignal>
#include <filesystem>
#include <print>
#include <vector>

SplitResults inference(Dataset& dataset, Network* network, DataSplit& split) {
    MetricsResult train_metrics, test_metrics;

    // Evaluate the model
    Matrix train_features = dataset.features(Eigen::placeholders::all, split.train_indices);
    Matrix train_labels = dataset.labels(Eigen::placeholders::all, split.train_indices);
    Matrix test_features = dataset.features(Eigen::placeholders::all, split.test_indices);
    Matrix test_labels = dataset.labels(Eigen::placeholders::all, split.test_indices);

    Matrix final_train_predictions = network->predict(train_features);
    Matrix final_test_predictions = network->predict(test_features);

    train_metrics.add(train_labels, final_train_predictions, network->getLossFunction());
    test_metrics.add(test_labels, final_test_predictions, network->getLossFunction());

    return {train_metrics, test_metrics};
}

void predict(Dataset& dataset, Network* network, Args& args) {
    std::vector<DataSplit> final_folds = DataSplit::split(dataset.num_samples, 5, args.train_ratio, args.shuffle);
    SplitResults results;
    for (size_t i = 0; i < final_folds.size(); ++i) {
        results.merge(inference(dataset, network, final_folds[i]));
    }

    results.average();
    bool is_regression = network->model.output_activation == ActivationType::LINEAR;
    std::println("\nTraining Set Metrics:");
    results.train_metrics.print(is_regression);
    std::println("Test Set Metrics:");
    results.test_metrics.print(is_regression);
}

int main(int argc, char* argv[]) {
    std::println("\n\n[Neural Network Training]");

    std::signal(SIGUSR1, handle_signal);
    Args args = Args::parse(argc, argv);

    MODEL_PATH = "artifacts/" + args.name;
    std::filesystem::create_directories(MODEL_PATH);

    // Set the random seed for reproducibility
    set_random_seed(args.seed);

    // Load the dataset
    Dataset dataset = Dataset::load(args.dataset_type, args.dataset_ratio);

    if (args.train) {
        // Prepare outer folds for cross-validation and load the grid search parameters
        std::vector<DataSplit> outer_folds = DataSplit::split(dataset.num_samples, args.outer_folds, args.train_ratio, args.shuffle);
        std::vector<Model> grid_search = Model::load_grid_search(args.model_file);
        std::map<int, Model> best_models;
        // Outer loop for cross-validation
        for (size_t i = 0; i < outer_folds.size(); ++i) {
            // Keep track of the best model for the current outer fold
            SplitResults best_model_performance;

            // Prepare inner folds for model selection, on training set of the current outer fold (train_valid)
            std::vector<DataSplit> inner_folds = DataSplit::split(outer_folds[i].train_indices.size(), args.inner_folds, args.train_ratio, args.shuffle);
            // Remap the inner fold indices to the original dataset indices
            for (auto& inner_fold : inner_folds) {
                for (int& relative_idx : inner_fold.train_indices) {
                    relative_idx = outer_folds[i].train_indices[relative_idx];
                }
                for (int& relative_idx : inner_fold.test_indices) {
                    relative_idx = outer_folds[i].train_indices[relative_idx];
                }
            }

            // Inner loop for model selection
            for (const auto& grid_model : grid_search) {
                SplitResults current_model_performance; // Metrics for the current model across inner folds
                // Loop through inner folds if model selection is required, otherwise train on the entire train_valid set
                for (size_t j = 0; j < inner_folds.size() && grid_search.size() > 1; ++j) {
                    Network network(grid_model, dataset.num_features, dataset.num_classes);                                // Create a new network for the current model configuration
                    SplitResults train_results = network.train(dataset, inner_folds[j], args.epochs, grid_model.id, i, j); // Train the model on the current inner fold
                    current_model_performance.merge(train_results);                                                        // Accumulate metrics for the current model across inner folds
                }

                current_model_performance.average(); // Average the metrics across inner folds for the current model to get a single performance metric

                // Compare the current model's performance with the best model's performance and update if it's better
                if (current_model_performance > best_model_performance) {
                    best_models.insert_or_assign(i, grid_model);
                    best_model_performance = current_model_performance;
                    best_models[i].epochs = best_model_performance.get_best_index() + 1; // Store the best epoch for the current outer fold
                }
            }

            // Re-train the best model of each outer fold on the train_valid set
            Network best_fold_network = Network(best_models[i], dataset.num_features, dataset.num_classes);
            best_fold_network.train(dataset, outer_folds[i], args.epochs, best_models[i].id, i, -1);
        }

        std::println("\n[Best Models Summary]");
        for (const auto& [outer_index, best_model] : best_models) {
            std::println(" • Fold {}: Model {} at epoch {}", outer_index, best_model.id, best_model.epochs);
        }
        if (args.dump) {
            std::print("\n[Dumping Best Models]");
            // Dump the best models trained on the entire dataset after cross-validation
            DataSplit full_split = DataSplit::split(dataset.num_samples, 1, 1.0, args.shuffle).front(); // Create a single split containing the entire dataset
            for (size_t i = 0; i < outer_folds.size(); ++i) {
                Network final_network(best_models[i], dataset.num_features, dataset.num_classes);
                final_network.train(dataset, full_split, best_models[i].epochs, best_models[i].id, i, -1, false);
                Serializer::dump_model(std::format("{}/outer{}.bin", MODEL_PATH, i), best_models[i], &final_network);
            }
        }
        return 0;
    }

    for (const auto& file : std::filesystem::directory_iterator(MODEL_PATH)) {
        if (file.path().extension() != ".bin") {
            continue;
        }
        // Load the final model and evaluate it on a dummy split of the dataset
        Network* network = Serializer::load_model(file.path());
        if (network == nullptr) {
            std::println(stderr, "Failed to load the model, skipping");
            continue;
        }
        predict(dataset, network, args);
        delete network;
    }
    return 0;
}