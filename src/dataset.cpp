#include "dataset.hpp"

#include "functions.hpp"

#include <fstream>
#include <iostream>

Matrix load_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::string line, cell;
    std::vector<Scalar> values;
    int rows = 0;
    int cols = 0;
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        int current_cols = 0;
        while (std::getline(lineStream, cell, ',')) {
            try {
                if (typeid(Scalar) == typeid(double)) {
                    values.push_back(std::stod(cell));
                } else {
                    values.push_back(std::stof(cell));
                }
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
    return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, cols);
}

Matrix load_mnist_images(const std::string& path, Scalar dataset_ratio) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    if (reverseInt(magic_number) != 2051) {
        throw std::runtime_error("Invalid MNIST image file");
    }

    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    num_images = static_cast<int>(reverseInt(num_images) * dataset_ratio);
    int image_size = reverseInt(num_rows) * reverseInt(num_cols);

    std::vector<unsigned char> raw_pixels(num_images * image_size);
    file.read((char*)raw_pixels.data(), raw_pixels.size());

    std::vector<Scalar> normalized_pixels(num_images * image_size);
    for (size_t i = 0; i < raw_pixels.size(); ++i) {
        normalized_pixels[i] = static_cast<Scalar>(raw_pixels[i]) / 255.0;
    }

    return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(normalized_pixels.data(), num_images, image_size);
}

Matrix load_mnist_labels(const std::string& path, Scalar dataset_ratio) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    int magic_number = 0, num_images = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    if (reverseInt(magic_number) != 2049) {
        throw std::runtime_error("Invalid MNIST label file");
    }

    file.read((char*)&num_images, sizeof(num_images));
    num_images = static_cast<int>(reverseInt(num_images) * dataset_ratio);

    std::vector<unsigned char> raw_labels(num_images);
    file.read((char*)raw_labels.data(), num_images);

    Matrix one_hot_labels = Matrix::Zero(num_images, 10);
    for (int i = 0; i < num_images; ++i) {
        one_hot_labels(i, static_cast<int>(raw_labels[i])) = 1.0;
    }

    return one_hot_labels;
}

SetIndices holdout_dataset(int num_samples, Scalar train_ratio) {
    int train_size = static_cast<int>(num_samples * train_ratio);
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), get_random_generator());

    SetIndices set_indices;
    set_indices.train_indices.assign(indices.begin(), indices.begin() + train_size);
    set_indices.test_indices.assign(indices.begin() + train_size, indices.end());

    return set_indices;
}

ModelSet load_dataset(DatasetType dataset_type, Scalar train_ratio, Scalar dataset_ratio) {
    switch (dataset_type) {
    case DatasetType::XOR: {
        Matrix xor_data = load_csv("dataset/xor/xor.csv").transpose();

        Matrix features = xor_data.topRows(xor_data.rows() - 1);
        Matrix labels = xor_data.bottomRows(1);

        Dataset train_set{features, labels};
        Dataset test_set{features, labels};

        return ModelSet{train_set, test_set};
    };
    case DatasetType::XOR_HOT: {
        Matrix xor_hot_data = load_csv("dataset/xor/xor_hot.csv").transpose();

        Matrix features = xor_hot_data.topRows(xor_hot_data.rows() - 2);
        Matrix labels = xor_hot_data.bottomRows(2);

        Dataset train_set{features, labels};
        Dataset test_set{features, labels};

        return ModelSet{train_set, test_set};
    };
    case DatasetType::MNIST: {
        Matrix train_features = load_mnist_images("dataset/mnist/train-images-idx3-ubyte", dataset_ratio).transpose();
        Matrix train_labels = load_mnist_labels("dataset/mnist/train-labels-idx1-ubyte", dataset_ratio).transpose();
        Matrix test_features = load_mnist_images("dataset/mnist/t10k-images-idx3-ubyte", dataset_ratio).transpose();
        Matrix test_labels = load_mnist_labels("dataset/mnist/t10k-labels-idx1-ubyte", dataset_ratio).transpose();

        Dataset train_set{train_features, train_labels};
        Dataset test_set{test_features, test_labels};

        return ModelSet{train_set, test_set};
    };
    default: {
        throw std::invalid_argument("Unsupported dataset type.");

        // Template for other datasets

        Matrix features, labels;
        Dataset dataset{features, labels};

        SetIndices set_indices = holdout_dataset(dataset.num_samples, train_ratio);

        Matrix train_features = dataset.features(Eigen::placeholders::all, set_indices.train_indices);
        Matrix train_labels = dataset.labels(Eigen::placeholders::all, set_indices.train_indices);

        Matrix test_features = dataset.features(Eigen::placeholders::all, set_indices.test_indices);
        Matrix test_labels = dataset.labels(Eigen::placeholders::all, set_indices.test_indices);

        Dataset train_set{train_features, train_labels};
        Dataset test_set{test_features, test_labels};

        return ModelSet{train_set, test_set};
    }
    }
}