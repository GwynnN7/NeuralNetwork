#include "dataset.hpp"

#include "types.hpp"
#include "utility.hpp"

#include <algorithm>
#include <format>
#include <fstream>
#include <istream>
#include <numeric>
#include <print>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Loader {
// MNIST utility functions for reading data
namespace MNIST {
constexpr int MNIST_IMAGE_MAGIC = 2051;
constexpr int MNIST_LABEL_MAGIC = 2049;
// Read exactly 'bytes' bytes from the input stream
void read_exact(std::istream& in, void* dest, std::size_t bytes, const std::string& path, const char* action) {
    in.read(reinterpret_cast<char*>(dest), static_cast<std::streamsize>(bytes));
    if (!in || in.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("Error processing " + path + " while reading " + action);
    }
}
// Read a big-endian integer from the input stream
int read_mnist_int(std::istream& in, const std::string& path, const char* action) {
    int value = 0;
    read_exact(in, &value, sizeof(value), path, action);
    return swap_endian(value);
}
// Take a subset of the total number of samples based on the dataset_ratio, ensuring at least one sample is returned
int sample_subset(int total, Scalar dataset_ratio) {
    if (total <= 0) {
        throw std::runtime_error("Invalid total number of samples: " + std::to_string(total));
    }
    const int scaled = static_cast<int>(total * dataset_ratio);
    return std::max(1, std::min(scaled, total));
}
// MNIST files are big-endian, need to swap on little-endian systems
std::int32_t swap_endian(std::int32_t value) {
    if constexpr (std::endian::native == std::endian::little) {
        return static_cast<std::int32_t>(std::byteswap(static_cast<std::uint32_t>(value)));
    }
    return value;
}
} // namespace MNIST

Matrix one_hot_encode(const Matrix& raw_labels, int num_classes) {
    // Initialize a zero matrix for one-hot encoding
    Matrix one_hot_labels = Matrix::Zero(raw_labels.size(), num_classes);
    for (Eigen::Index i = 0; i < raw_labels.size(); ++i) {
        const int label = static_cast<int>(raw_labels(i));
        if (label < 0 || label >= num_classes) {
            throw std::out_of_range("Label " + std::to_string(label) + " is out of range for one-hot encoding with " + std::to_string(num_classes) + " classes");
        }
        // Set the corresponding index to 1 for one-hot encoding
        one_hot_labels(i, label) = 1.0;
    }
    return one_hot_labels;
}

Matrix load_mnist_images(const std::string& path, Scalar dataset_ratio) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Read the magic number and validate it
    if (MNIST::read_mnist_int(file, path, "magic number") != MNIST::MNIST_IMAGE_MAGIC) {
        throw std::runtime_error("Invalid MNIST image file: " + path);
    }

    // Read the number of images, rows, and columns
    const int total_images = MNIST::read_mnist_int(file, path, "image count");
    const int num_rows = MNIST::read_mnist_int(file, path, "row count");
    const int num_cols = MNIST::read_mnist_int(file, path, "column count");
    if (num_rows <= 0 || num_cols <= 0) {
        throw std::runtime_error("Invalid MNIST image dimensions in " + path);
    }

    // Define a subset of images based on the dataset ratio
    const int num_images = MNIST::sample_subset(total_images, dataset_ratio);
    const int image_size = num_rows * num_cols;

    // Read the raw pixel data
    std::vector<unsigned char> raw_pixels(static_cast<std::size_t>(num_images) * image_size);
    MNIST::read_exact(file, raw_pixels.data(), raw_pixels.size(), path, "pixel data");

    // Normalize pixel values to the range [0, 1] and store them in a matrix
    Matrix normalized_images(num_images, image_size);
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            normalized_images(i, j) = static_cast<Scalar>(raw_pixels[static_cast<std::size_t>(i) * image_size + j]) / Scalar(255);
        }
    }
    return normalized_images;
}

Matrix load_mnist_labels(const std::string& path, Scalar dataset_ratio) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Read the magic number and validate it
    if (MNIST::read_mnist_int(file, path, "magic number") != MNIST::MNIST_LABEL_MAGIC) {
        throw std::runtime_error("Invalid MNIST label file: " + path);
    }

    // Read the number of labels and define a subset based on the dataset ratio
    const int total_labels = MNIST::read_mnist_int(file, path, "label count");
    const int num_labels = MNIST::sample_subset(total_labels, dataset_ratio);

    // Read the raw label data
    std::vector<unsigned char> raw_labels(static_cast<std::size_t>(num_labels));
    MNIST::read_exact(file, raw_labels.data(), raw_labels.size(), path, "label data");

    // Convert the unsigned char vector into an Eigen Scalar Matrix
    Matrix scalar_labels(num_labels, 1);
    for (int i = 0; i < num_labels; ++i) {
        scalar_labels(i, 0) = static_cast<Scalar>(raw_labels[i]);
    }

    // Convert to one-hot encoding
    return one_hot_encode(scalar_labels, 10);
}

Matrix load_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::string line, cell;
    std::vector<Scalar> values;
    int rows = 0, cols = 0;
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        int current_cols = 0;
        // Read each cell in the line (split by commas), convert to Scalar, and store in values
        while (std::getline(lineStream, cell, ',')) {
            try {
                values.push_back(static_cast<Scalar>(std::stod(cell)));
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid number in dataset: " + cell);
            }
            current_cols++;
        }
        if (cols == 0) {
            cols = current_cols;
        } else if (cols != current_cols) {
            throw std::runtime_error("Invalid number of columns in dataset.");
        }
        rows++;
    }
    file.close();

    auto row_major_map = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, cols);
    // Copy the mapped data into a new ColMajor matrix to ensure compatibility with the rest of the framework
    Matrix col_major = row_major_map;
    return col_major;
}

Matrix load_monk(const std::string& filename, Scalar dataset_ratio) {
    std::ifstream file(filename);
    std::string line, cell;
    std::vector<Scalar> values;
    int rows = 0;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Define the sizes of the one-hot encoded features for each categorical feature
    std::vector<int> feature_hot_sizes = {3, 3, 2, 3, 4, 2};

    // Expected cols have to be 1 label column + one column per one-hot slot
    const int expected_cols = 1 + std::accumulate(feature_hot_sizes.begin(), feature_hot_sizes.end(), 0);

    while (std::getline(file, line)) {
        // Skip blank and whitespace lines
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        std::stringstream lineStream(line);
        int current_cols = 0;

        while (lineStream >> cell) {
            if (current_cols == 7) { // Skip the last column (tag)
                break;
            }
            try {
                Scalar val = static_cast<Scalar>(std::stod(cell));
                if (current_cols == 0) {
                    values.push_back(val);
                } else {
                    // One-hot encode the categorical features based on their respective sizes
                    int num_features = feature_hot_sizes[current_cols - 1];
                    if (val < 1 || static_cast<int>(val) > num_features) {
                        throw std::runtime_error("MONK attribute " + std::to_string(current_cols) + " out of range in " + filename + ": " + cell);
                    }
                    for (int i = 1; i <= num_features; ++i) {
                        values.push_back((i == static_cast<int>(val)) ? 1.0 : 0.0);
                    }
                }
            } catch (const std::invalid_argument&) {
                throw std::runtime_error("Invalid number in dataset: " + cell);
            }
            current_cols++;
        }
        if (current_cols != 7) {
            throw std::runtime_error("Invalid number of columns in MONK dataset: " + filename);
        }
        rows++;
    }
    file.close();

    if (rows == 0 || static_cast<int>(values.size()) != rows * expected_cols) {
        throw std::runtime_error("Error loading MONK dataset: " + filename);
    }

    const int kept_rows = std::max(1, std::min(rows, static_cast<int>(rows * dataset_ratio)));
    auto row_major_map = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, expected_cols).topRows(kept_rows);
    // Copy the mapped data into a new ColMajor matrix to ensure compatibility with the rest of the framework
    Matrix col_major = row_major_map;
    return col_major;
}
} // namespace Loader

void Dataset::print_info() const {
    std::println("\nDataset Info:");
    std::println("{:<25}{}", " • Type:", Maps::dataset_to_str.at(type));
    std::println("{:<25}{}", " • Task:", Maps::task_to_str.at(task));
    std::println("{:<25}{}", " • Samples:", num_samples);
    std::println("{:<25}{}", " • Features:", num_features);
    std::println("{:<25}{}", " • Classes:", num_classes);
}

Dataset Dataset::load(DatasetType dataset_type, Scalar dataset_ratio) {
    if (!(dataset_ratio > 0) || dataset_ratio > 1) {
        throw std::invalid_argument("dataset_ratio must be in (0, 1]");
    }
    switch (dataset_type) {
    case DatasetType::XOR:
    case DatasetType::XOR_HOT: {
        Matrix xor_data = Loader::load_csv("dataset/xor/xor.csv").transpose();

        // Split the data into features and labels
        Matrix features = xor_data.topRows(xor_data.rows() - 1);
        Matrix labels = dataset_type == DatasetType::XOR ? xor_data.bottomRows(1).eval() : Loader::one_hot_encode(xor_data.bottomRows(1), 2).transpose();

        // Replicate the dataset to create a larger dataset for training and testing (task set to regression for XOR and classification for XOR_HOT)
        return Dataset{dataset_type, dataset_type == DatasetType::XOR ? TaskType::REGRESSION : TaskType::CLASSIFICATION, features.replicate(1, 100), labels.replicate(1, 100)};
    };
    case DatasetType::MONK_1:
    case DatasetType::MONK_1_HOT:
    case DatasetType::MONK_2:
    case DatasetType::MONK_2_HOT:
    case DatasetType::MONK_3:
    case DatasetType::MONK_3_HOT: {
        const char monk_dataset_number = Maps::dataset_to_str.at(dataset_type)[5]; // Extract the dataset number from the enum name
        Matrix monk_train = Loader::load_monk(std::format("dataset/monk/monks-{}.train", monk_dataset_number), dataset_ratio).transpose();
        Matrix monk_test = Loader::load_monk(std::format("dataset/monk/monks-{}.test", monk_dataset_number), dataset_ratio).transpose();
        Matrix monk_data(monk_train.rows(), monk_train.cols() + monk_test.cols());
        monk_data << monk_train, monk_test;

        Matrix features = monk_data.bottomRows(monk_data.rows() - 1);
        const bool is_hot_encoded = (dataset_type == DatasetType::MONK_1_HOT || dataset_type == DatasetType::MONK_2_HOT || dataset_type == DatasetType::MONK_3_HOT);
        Matrix labels = !is_hot_encoded ? monk_data.topRows(1).eval() : Loader::one_hot_encode(monk_data.topRows(1), 2).transpose();

        return Dataset{dataset_type, TaskType::CLASSIFICATION, std::move(features), std::move(labels), static_cast<int>(monk_train.cols())};
    };
    case DatasetType::MNIST: {
        Matrix train_features = Loader::load_mnist_images("dataset/mnist/train-images-idx3-ubyte", dataset_ratio).transpose();
        Matrix train_labels = Loader::load_mnist_labels("dataset/mnist/train-labels-idx1-ubyte", dataset_ratio).transpose();
        Matrix test_features = Loader::load_mnist_images("dataset/mnist/t10k-images-idx3-ubyte", dataset_ratio).transpose();
        Matrix test_labels = Loader::load_mnist_labels("dataset/mnist/t10k-labels-idx1-ubyte", dataset_ratio).transpose();

        // Combine training and testing datasets into a single dataset for cross-validation
        Matrix all_features(train_features.rows(), train_features.cols() + test_features.cols());
        Matrix all_labels(train_labels.rows(), train_labels.cols() + test_labels.cols());

        all_features << train_features, test_features;
        all_labels << train_labels, test_labels;

        // Return the combined dataset with task set to classification for MNIST
        const int mnist_train_samples = static_cast<int>(train_features.cols());
        return Dataset{DatasetType::MNIST, TaskType::CLASSIFICATION, std::move(all_features), std::move(all_labels), mnist_train_samples};
    };
    default: {
        throw std::invalid_argument("Unsupported dataset type");
    }
    }
}

std::vector<DataSplit> DataSplit::split(int k, Scalar train_ratio, int train_samples, int num_samples, bool shuffle, std::optional<unsigned int> seed) {
    if (k < 0) {
        throw std::invalid_argument("Number of folds must be non-negative");
    }
    if (k == 0) {
        return std::vector<DataSplit>();
    }
    if (num_samples < 2) {
        throw std::invalid_argument("Need at least 2 samples to split the dataset");
    }
    if (!(train_ratio > 0) || train_ratio >= 1) {
        throw std::invalid_argument("Argument train_ratio must be in (0, 1)");
    }

    bool holdout_split = (k == 1 || k > num_samples);

    // If in holdout split, and the dataset has a predefined number of training samples, use that number instead of calculating it from the train_ratio
    // Only used for outer cross-validation. In inner cross-validation folds, the argument 'train_samples' will be passed as 0
    bool use_train_samples = (train_samples > 0 && train_samples < num_samples) && holdout_split;

    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    // Don't shuffle the indices if using the original training samples for outer holdout split, to maintain the samples in the splits
    if (shuffle && !use_train_samples) {
        // Shuffle the indices to ensure random distribution of samples across folds
        if (seed.has_value()) {
            std::mt19937 local_generator(*seed);
            std::ranges::shuffle(indices, local_generator);
        } else {
            std::ranges::shuffle(indices, get_random_generator());
        }
    }

    // Holdout split
    if (holdout_split) {
        std::vector<DataSplit> folds(1);
        int train_size = static_cast<int>(use_train_samples ? train_samples : num_samples * train_ratio);
        // Avoid empty sets by clamping the train_size
        train_size = std::clamp(train_size, 1, num_samples - 1);
        // Assign the first 'train_size' samples to the training set and the rest to the test set
        folds[0].train_indices.assign(indices.begin(), indices.begin() + train_size);
        folds[0].test_indices.assign(indices.begin() + train_size, indices.end());
        return folds;
    }

    std::vector<DataSplit> folds(k);
    // Calculate the size of each fold and distribute the remaining samples on the first few folds
    int fold_size = num_samples / k;
    int remaining_samples = num_samples % k;

    int current_start = 0;
    for (int i = 0; i < k; ++i) {
        // Add one more sample to each fold until the remaining samples are distributed
        int current_fold_size = fold_size + (i < remaining_samples ? 1 : 0);
        int end = current_start + current_fold_size;

        // Assign test set
        folds[i].test_indices.assign(indices.begin() + current_start, indices.begin() + end);

        // Assign train set (samples before + samples after)
        folds[i].train_indices.assign(indices.begin(), indices.begin() + current_start);
        folds[i].train_indices.insert(folds[i].train_indices.end(), indices.begin() + end, indices.end());

        // Move the sliding window forward
        current_start = end;
    }

    return folds;
}