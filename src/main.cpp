#include "cli.hpp"
#include "dataset.hpp"
#include "datasplit.hpp"
#include "model.hpp"
#include "network.hpp"
#include "selection.hpp"
#include "summary.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <algorithm>
#include <csignal>
#include <filesystem>
#include <format>
#include <print>
#include <vector>

void train(const Dataset& dataset, const Args& args) {
    // Prepare the main split for final evaluation and dumping of best models
    // Every split for this run comes from here, so the dataset and the options behind them stay visible
    const Splitter splitter{dataset, args};
    const DataSplit final_split = splitter.final_split();

    // Load the grid search parameters and assign the task type to the model based on the dataset
    std::vector<Model> grid_search = Model::load_grid_search(args.grid_file);
    std::for_each(grid_search.begin(), grid_search.end(), [&dataset](Model& model) { model.task = dataset.task; });

    // Initialize outer folds variables
    const std::vector<DataSplit> outer_folds = splitter.get(args.outer_folds);
    std::vector<Model> best_models;
    best_models.reserve(outer_folds.size());
    // Initialize a vector to store the summary of the best model of each outer fold
    std::vector<SplitSummary> assessment_summaries(outer_folds.size());

    // Outer loop for cross-validation
    for (size_t i = 0; i < outer_folds.size(); ++i) {
        const int outer_index = static_cast<int>(i);

        // Prepare inner folds for model selection, on training set of the current outer fold (train_valid)
        std::vector<DataSplit> inner_folds = splitter.get(args.inner_folds, &outer_folds[i].train_indices);
        bool in_model_selection = grid_search.size() > 1;

        // Variables to track the best model and its score across inner folds
        size_t best_model_index = 0;
        SelectionScore best_score;

        // Inner loop for model selection, skipped when there is nothing to select
        for (size_t m = 0; in_model_selection && m < grid_search.size(); ++m) {
            // Get next model and print its configuration
            const Model& grid_model = grid_search[m];
            grid_model.print();

            // Run the model on each inner fold and collect scores for selection and summary for statistics
            std::vector<SelectionScore> model_scores;
            SplitSummary model_summary;
            for (size_t j = 0; j < inner_folds.size(); ++j) {
                for (int t = 0; t < args.trials; ++t) { // Loop over trials for the current inner fold
                    set_trial_seed(args.seed, t, j);    // Seed the trial generator for reproducibility

                    Network network(grid_model, dataset.num_features, dataset.num_classes);

                    const TrainContext ctx{
                        .trial = t,
                        .epochs = args.epochs,
                        .patience = args.patience,
                        .warmup = args.warmup,
                        .model_id = grid_model.id,
                        .outer_index = outer_index,
                        .inner_index = static_cast<int>(j),
                        .in_model_selection = true,
                        .logging = true,
                    };

                    // Train the model on the current inner fold
                    const RunCurves run = network.train(dataset, inner_folds[j], ctx);
                    model_scores.push_back(SelectionScore::from_run(run));
                    model_summary.add_trial(run);
                }
            }

            model_summary.print(dataset.task == TaskType::CLASSIFICATION, "Validation");

            // Calculate the average score of the model across all inner folds and trials
            const SelectionScore average_score = SelectionScore::average_scores(model_scores);
            // Keep the model with the best final score
            if (average_score.valid() && (m == 0 || !best_score.valid() || average_score < best_score)) {
                best_model_index = m;
                best_score = average_score;
            }
        }

        best_models.push_back(grid_search[best_model_index]);
        if (in_model_selection) {
            std::println("\nRetraining the best model for Outer Fold {}: Model {}", outer_index, best_models[i].id);
        }
        best_models[i].print();

        // (Re)train the best model on the entire training set of the outer fold
        for (int t = 0; t < args.trials; ++t) {
            set_trial_seed(args.seed, t, -1);

            Network best_fold_network(best_models[i], dataset.num_features, dataset.num_classes);
            const TrainContext ctx{
                .trial = t,
                .epochs = args.epochs,
                .patience = args.patience,
                .warmup = args.warmup,
                .model_id = best_models[i].id,
                .outer_index = outer_index,
                .inner_index = -1,
                .in_model_selection = false,
                .logging = true,
            };
            const RunCurves run = best_fold_network.train(dataset, outer_folds[i], ctx);
            assessment_summaries[i].add_trial(run);
        }
    }

    // Print the summary of the best models found for each outer fold
    std::println("\n[Best Models Summary]");
    const bool track_accuracy = (dataset.task == TaskType::CLASSIFICATION);
    for (size_t i = 0; i < best_models.size(); ++i) {
        std::println("\n • Fold {}: Model {}", i, best_models[i].id);
        assessment_summaries[i].print(track_accuracy, "Test");
    }

    // Dump the best models found with cross-validation, trained on the original train/test split (if applicable)
    if (args.dump) {
        std::print("\n[Dumping Best Models]");
        for (size_t i = 0; i < best_models.size(); ++i) {
            set_trial_seed(args.seed, 0, -1);

            Network final_network(best_models[i], dataset.num_features, dataset.num_classes);
            const TrainContext ctx{
                .epochs = args.epochs,
                .patience = args.patience,
                .warmup = args.warmup,
                .model_id = best_models[i].id,
                .outer_index = static_cast<int>(i),
                .inner_index = -1,
                .in_model_selection = false,
                .logging = false,
            };
            final_network.train(dataset, final_split, ctx);
            Serializer::dump_model(std::format("{}/outer{}.bin", MODEL_PATH, i), best_models[i], final_network);
        }
    }
}

void test(const Dataset& dataset, const Args& args) {
    if (!std::filesystem::is_directory(MODEL_PATH)) {
        throw std::runtime_error("No artifact directory to load models from: " + MODEL_PATH);
    }

    // Collect the .bin model files and sort them
    std::vector<std::filesystem::path> models;
    for (const auto& file : std::filesystem::directory_iterator(MODEL_PATH)) {
        if (file.path().extension() == ".bin") {
            models.push_back(file.path());
        }
    }
    std::ranges::sort(models);

    if (models.empty()) {
        throw std::runtime_error("No .bin model files found in " + MODEL_PATH + " (run with --train --dump first)");
    }

    // Get the same split the dumped models were trained on
    const Splitter splitter{dataset, args};
    const DataSplit final_split = splitter.final_split();

    // Extract training and testing features and labels for the split
    const Matrix train_features = dataset.features(Eigen::placeholders::all, final_split.train_indices);
    const Matrix train_labels = dataset.labels(Eigen::placeholders::all, final_split.train_indices);
    const Matrix test_features = dataset.features(Eigen::placeholders::all, final_split.test_indices);
    const Matrix test_labels = dataset.labels(Eigen::placeholders::all, final_split.test_indices);

    // Load the best models trained and evaluate them
    for (const auto& file : models) {
        std::unique_ptr<Network> network = Serializer::load_model(file, dataset);
        if (network == nullptr) {
            std::println(stderr, "Failed to load the model, skipping");
            continue;
        }

        // Evaluate the model
        const Matrix final_train_predictions = network->predict(train_features);
        const Matrix final_test_predictions = network->predict(test_features);

        // A loaded model is evaluated once, so there is no curve and no spread to report here
        const bool track_accuracy = (network->model.task == TaskType::CLASSIFICATION);
        const LossType loss = network->model.loss_type;

        SplitSummary summary;
        summary.add_trial(Metrics::evaluate(train_labels, final_train_predictions, loss, track_accuracy),
                          Metrics::evaluate(test_labels, final_test_predictions, loss, track_accuracy));
        summary.print(track_accuracy, "Test");
    }
}

int main(int argc, char* argv[]) {
    std::println("\n\n[Neural Network Framework]");

    std::signal(SIGUSR1, handle_signal);

    try {
        Args args = Args::parse(argc, argv);

        MODEL_PATH = "artifacts/" + args.name;
        std::filesystem::create_directories(MODEL_PATH);

        // Set the random seed for reproducibility
        set_split_seed(static_cast<unsigned int>(args.seed));

        // Load the dataset
        Dataset dataset = Dataset::load(args.dataset_type, args.dataset_ratio);
        dataset.print_info();

        if (args.train) {
            train(dataset, args);
        } else {
            test(dataset, args);
        }
    } catch (const std::exception& e) {
        std::println(stderr, "\n[Fatal Error] {}", e.what());
        return 1;
    }

    return 0;
}
