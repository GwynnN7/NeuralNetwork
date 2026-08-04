#include "dataset.hpp"

#include "utility.hpp"

#include <algorithm>
#include <bit>
#include <fstream>
#include <iostream>
#include <vector>

namespace Loader {

#define MNIST_IMAGE_MAGIC 2051
#define MNIST_LABEL_MAGIC 2049

Matrix one_hot_encode(const Matrix& raw_labels, int num_classes) {
    Matrix one_hot_labels = Matrix::Zero(raw_labels.size(), num_classes);
    for (int i = 0; i < raw_labels.size(); ++i) {
        try {
            int label = static_cast<int>(raw_labels(i));
            if (label < 0 || label >= num_classes) {
                throw std::out_of_range("Label value out of range for one-hot encoding");
            }
            one_hot_labels(i, label) = 1.0;
        } catch (const std::exception& e) {
            std::cerr << "Error encoding label at index " << i << ": " << e.what() << std::endl;
            throw;
        }
    }
    return one_hot_labels;
}
Matrix load_mnist_images(const std::string& path, Scalar dataset_ratio) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;

    // Read the magic number and validate it
    file.read((char*)&magic_number, sizeof(magic_number));
    if (std::byteswap(magic_number) != MNIST_IMAGE_MAGIC) {
        throw std::runtime_error("Invalid MNIST image file");
    }

    // Read the number of images, rows, and columns
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    // Define a subset of images based on the dataset ratio
    num_images = static_cast<int>(std::byteswap(num_images) * dataset_ratio);
    int image_size = std::byteswap(num_rows) * std::byteswap(num_cols);

    // Read the raw pixel data
    std::vector<unsigned char> raw_pixels(num_images * image_size);
    file.read((char*)raw_pixels.data(), raw_pixels.size());

    // Normalize pixel values to the range [0, 1] and store them in a matrix
    std::vector<Scalar> normalized_pixels(num_images * image_size);
    for (size_t i = 0; i < raw_pixels.size(); ++i) {
        normalized_pixels[i] = static_cast<Scalar>(raw_pixels[i]) / 255.0;
    }

    // Return the data as an Eigen matrix with row-major storage to match the expected input format for the framework
    return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(normalized_pixels.data(), num_images, image_size);
}

Matrix load_mnist_labels(const std::string& path, Scalar dataset_ratio) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    int magic_number = 0, num_images = 0;

    // Read the magic number and validate it
    file.read((char*)&magic_number, sizeof(magic_number));
    if (std::byteswap(magic_number) != MNIST_LABEL_MAGIC) {
        throw std::runtime_error("Invalid MNIST label file");
    }

    // Read the number of images and define a subset based on the dataset ratio
    file.read((char*)&num_images, sizeof(num_images));
    num_images = static_cast<int>(std::byteswap(num_images) * dataset_ratio);

    // Read the raw label data
    std::vector<unsigned char> raw_labels(num_images);
    file.read((char*)raw_labels.data(), num_images);

    // Convert the unsigned char vector into an Eigen Scalar Matrix
    Matrix scalar_labels(num_images, 1);
    for (int i = 0; i < num_images; ++i) {
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
            throw std::runtime_error("Inconsistent number of columns in dataset.");
        }
        rows++;
    }
    file.close();
    // Return the data as an Eigen matrix with row-major storage to match the expected input format for the framework
    return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, cols);
}

Matrix load_monk(const std::string& filename, Scalar dataset_ratio) {
    std::ifstream file(filename);
    std::string line, cell;
    std::vector<Scalar> values;
    int rows = 0;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<int> feature_hot_sizes = {2, 3, 3, 2, 3, 4, 2};

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream lineStream(line);
        int current_cols = 0;

        while (lineStream >> cell) {
            if (current_cols == 7) { // Skip the last column (name/ID)
                break;
            }
            try {
                Scalar val = std::stod(cell);
                if (current_cols == 0) {
                    values.push_back(val);
                } else {
                    int num_features = feature_hot_sizes[current_cols];
                    for (int i = 1; i <= num_features; ++i) {
                        values.push_back((i == static_cast<int>(val)) ? 1.0 : 0.0);
                    }
                }
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid number in dataset: " + cell);
            }
            current_cols++;
        }
        rows++;
    }
    file.close();

    int cols = values.size() / rows;
    return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, cols).topRows(static_cast<int>(rows * dataset_ratio));
}
} // namespace Loader

Dataset Dataset::load(DatasetType dataset_type, Scalar dataset_ratio) {
    switch (dataset_type) {
    case DatasetType::XOR: {
        Matrix xor_data = Loader::load_csv("dataset/xor/xor.csv").transpose();

        // Split the data into features and labels
        Matrix features = xor_data.topRows(xor_data.rows() - 1);
        Matrix labels = xor_data.bottomRows(1);

        // Replicate the dataset to create a larger dataset for training and testing (task set to regression for XOR)
        Dataset dataset{DatasetType::XOR, TaskType::REGRESSION, features.replicate(1, 100), labels.replicate(1, 100)};
        return dataset;
    };
    case DatasetType::XOR_HOT: {
        Matrix xor_data = Loader::load_csv("dataset/xor/xor.csv").transpose();

        Matrix features = xor_data.topRows(xor_data.rows() - 1);
        Matrix labels = Loader::one_hot_encode(xor_data.bottomRows(1), 2).transpose();

        return Dataset{DatasetType::XOR_HOT, TaskType::CLASSIFICATION, features.replicate(1, 100), labels.replicate(1, 100)};
    };
    case DatasetType::MONK_1: {
        Matrix monk_train = Loader::load_monk("dataset/monk/monks-1.train", dataset_ratio).transpose();
        Matrix monk_test = Loader::load_monk("dataset/monk/monks-1.test", dataset_ratio).transpose();
        Matrix monk_data(monk_train.rows(), monk_train.cols() + monk_test.cols());
        monk_data << monk_train, monk_test;

        Matrix features = monk_data.bottomRows(monk_data.rows() - 1);
        Matrix labels = monk_data.topRows(1);

        return Dataset{DatasetType::MONK_1, TaskType::CLASSIFICATION, features, labels, static_cast<int>(monk_train.cols())};
    };
    case DatasetType::MONK_1_HOT: {
        Matrix monk_train = Loader::load_monk("dataset/monk/monks-1.train", dataset_ratio).transpose();
        Matrix monk_test = Loader::load_monk("dataset/monk/monks-1.test", dataset_ratio).transpose();
        Matrix monk_data(monk_train.rows(), monk_train.cols() + monk_test.cols());
        monk_data << monk_train, monk_test;

        Matrix features = monk_data.bottomRows(monk_data.rows() - 1);
        Matrix labels = Loader::one_hot_encode(monk_data.topRows(1), 2).transpose();

        return Dataset{DatasetType::MONK_1_HOT, TaskType::CLASSIFICATION, features, labels, static_cast<int>(monk_train.cols())};
    };
    case DatasetType::MONK_2: {
        Matrix monk_train = Loader::load_monk("dataset/monk/monks-2.train", dataset_ratio).transpose();
        Matrix monk_test = Loader::load_monk("dataset/monk/monks-2.test", dataset_ratio).transpose();
        Matrix monk_data(monk_train.rows(), monk_train.cols() + monk_test.cols());
        monk_data << monk_train, monk_test;

        Matrix features = monk_data.bottomRows(monk_data.rows() - 1);
        Matrix labels = monk_data.topRows(1);

        return Dataset{DatasetType::MONK_2, TaskType::CLASSIFICATION, features, labels, static_cast<int>(monk_train.cols())};
    };
    case DatasetType::MONK_2_HOT: {
        Matrix monk_train = Loader::load_monk("dataset/monk/monks-2.train", dataset_ratio).transpose();
        Matrix monk_test = Loader::load_monk("dataset/monk/monks-2.test", dataset_ratio).transpose();
        Matrix monk_data(monk_train.rows(), monk_train.cols() + monk_test.cols());
        monk_data << monk_train, monk_test;

        Matrix features = monk_data.bottomRows(monk_data.rows() - 1);
        Matrix labels = Loader::one_hot_encode(monk_data.topRows(1), 2).transpose();

        return Dataset{DatasetType::MONK_2_HOT, TaskType::CLASSIFICATION, features, labels, static_cast<int>(monk_train.cols())};
    };
    case DatasetType::MONK_3: {
        Matrix monk_train = Loader::load_monk("dataset/monk/monks-3.train", dataset_ratio).transpose();
        Matrix monk_test = Loader::load_monk("dataset/monk/monks-3.test", dataset_ratio).transpose();
        Matrix monk_data(monk_train.rows(), monk_train.cols() + monk_test.cols());
        monk_data << monk_train, monk_test;

        Matrix features = monk_data.bottomRows(monk_data.rows() - 1);
        Matrix labels = monk_data.topRows(1);

        return Dataset{DatasetType::MONK_3, TaskType::CLASSIFICATION, features, labels, static_cast<int>(monk_train.cols())};
    };
    case DatasetType::MONK_3_HOT: {
        Matrix monk_train = Loader::load_monk("dataset/monk/monks-3.train", dataset_ratio).transpose();
        Matrix monk_test = Loader::load_monk("dataset/monk/monks-3.test", dataset_ratio).transpose();
        Matrix monk_data(monk_train.rows(), monk_train.cols() + monk_test.cols());
        monk_data << monk_train, monk_test;

        Matrix features = monk_data.bottomRows(monk_data.rows() - 1);
        Matrix labels = Loader::one_hot_encode(monk_data.topRows(1), 2).transpose();

        return Dataset{DatasetType::MONK_3_HOT, TaskType::CLASSIFICATION, features, labels, static_cast<int>(monk_train.cols())};
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
        return Dataset{DatasetType::MNIST, TaskType::CLASSIFICATION, all_features, all_labels, static_cast<int>(train_features.cols())};
    };
    default: {
        throw std::invalid_argument("Unsupported dataset type");
    }
    }
}

std::vector<DataSplit> DataSplit::split(int num_samples, int k, Scalar train_ratio, int original_train_samples, bool shuffle) {
    if (k == 0) {
        return std::vector<DataSplit>();
    }

    bool holdout_split = (k == 1 || k > num_samples);

    // If in holdout split, and the dataset has a predefined number of training samples, use that number instead of calculating it from the train_ratio
    // Only used for outer cross-validation. In inner cross-validation folds, the argument 'original_train_samples' will be passed as 0
    bool use_original_train_samples = (original_train_samples > 0 && original_train_samples < num_samples) && holdout_split;

    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    // Don't shuffle the indices if using the original training samples for outer holdout split, to maintain the samples in the splits
    if (shuffle && !use_original_train_samples) {
        // Shuffle the indices to ensure random distribution of samples across folds
        std::ranges::shuffle(indices, get_random_generator());
    }

    // Holdout split
    if (holdout_split) {
        std::vector<DataSplit> folds(1);
        int train_size = static_cast<int>(use_original_train_samples ? original_train_samples : num_samples * train_ratio);
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