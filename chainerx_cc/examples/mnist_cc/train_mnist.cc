#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward.h"
#include "chainerx/device_id.h"
#include "chainerx/dtype.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/math.h"
#include "chainerx/shape.h"
#include "chainerx/slice.h"
#include "mnist.h"

namespace chx = chainerx;

chx::Array GetRandomArray(std::mt19937& gen, std::normal_distribution<float>& dist, const chx::Shape& shape) {
    int64_t n = shape.GetTotalSize();
    std::shared_ptr<float> data{new float[n], std::default_delete<float[]>{}};
    std::generate_n(data.get(), n, [&dist, &gen]() { return dist(gen); });
    return chx::FromContiguousHostData(shape, chx::TypeToDtype<float>, static_cast<std::shared_ptr<void>>(data), chx::GetDefaultDevice());
}

class Model {
public:
    Model(int64_t n_in, int64_t n_hidden, int64_t n_out, int64_t n_layers)
        : n_in_{n_in}, n_hidden_{n_hidden}, n_out_{n_out}, n_layers_{n_layers} {};

    chx::Array operator()(const chx::Array& x) {
        chx::Array h = x;
        for (int64_t i = 0; i < n_layers_; ++i) {
            h = chx::Dot(h, params_[i * 2]) + params_[i * 2 + 1];
            if (i != n_layers_ - 1) {
                h = chx::Maximum(0, h);
            }
        }
        return h;
    }

    const std::vector<chx::Array>& params() { return params_; }

    void Initialize(std::mt19937& gen, std::normal_distribution<float>& dist) {
        params_.clear();

        for (int64_t i = 0; i < n_layers_; ++i) {
            int64_t n_in = i == 0 ? n_in_ : n_hidden_;
            int64_t n_out = i == n_layers_ - 1 ? n_out_ : n_hidden_;
            params_.emplace_back(GetRandomArray(gen, dist, {n_in, n_out}));
            params_.emplace_back(chx::Zeros({n_out}, chx::Dtype::kFloat32));
        }

        for (const chx::Array& param : params_) {
            param.RequireGrad();
        }
    }

private:
    int64_t n_in_;
    int64_t n_hidden_;
    int64_t n_out_;
    int64_t n_layers_;
    std::vector<chx::Array> params_;
};

chx::Array SoftmaxCrossEntropy(const chx::Array& y, const chx::Array& t, bool normalize = true) {
    chx::Array score = chx::LogSoftmax(y, 1);
    chx::Array mask = (t.At({chx::Slice{}, chx::NewAxis{}}) == chx::Arange(score.shape()[1], t.dtype())).AsType(score.dtype());
    if (normalize) {
        return -(score * mask).Sum() / y.shape()[0];
    }
    return -(score * mask).Sum();
}

void Run(int64_t epochs, int64_t batch_size, int64_t n_hidden, int64_t n_layers, float lr, const std::string& mnist_root) {
    // Read the MNIST dataset.
    chx::Array train_x = ReadMnistImages(mnist_root + "train-images-idx3-ubyte");
    chx::Array train_t = ReadMnistLabels(mnist_root + "train-labels-idx1-ubyte");
    chx::Array test_x = ReadMnistImages(mnist_root + "t10k-images-idx3-ubyte");
    chx::Array test_t = ReadMnistLabels(mnist_root + "t10k-labels-idx1-ubyte");

    train_x = train_x.AsType(chx::Dtype::kFloat32) / 255.f;
    train_t = train_t.AsType(chx::Dtype::kInt32);
    test_x = test_x.AsType(chx::Dtype::kFloat32) / 255.f;
    test_t = test_t.AsType(chx::Dtype::kInt32);

    int64_t n_train = train_x.shape().front();
    int64_t n_test = test_x.shape().front();
    chx::Array train_indices = chx::Arange(n_train, chx::Dtype::kInt64);

    // Initialize the model with random parameters.
    Model model{train_x.shape()[1], n_hidden, 10, n_layers};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> dist{0.f, 0.05f};
    model.Initialize(gen, dist);

    auto start = std::chrono::high_resolution_clock::now();

    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        // The underlying data of the indices' Array is contiguous so we can use std::shuffle to randomize the order.
        assert(train_indices.IsContiguous());
        std::shuffle(
                reinterpret_cast<int64_t*>(train_indices.raw_data()), reinterpret_cast<int64_t*>(train_indices.raw_data()) + n_train, gen);
        for (int64_t i = 0; i < n_train; i += batch_size) {
            chx::Array indices = train_indices.At({chx::Slice{i, i + batch_size}});
            chx::Array x = train_x.Take(indices, 0);
            chx::Array t = train_t.Take(indices, 0);

            chx::Backward(SoftmaxCrossEntropy(model(x), t));

            // Vanilla SGD.
            for (const chx::Array& param : model.params()) {
                chx::Array p = param.AsGradStopped();
                p -= param.GetGrad()->AsGradStopped() * lr;
                param.ClearGrad();
            }
        }

        // Evaluate.
        {
            chx::NoBackpropModeScope scope{};

            chx::Array loss = chx::Zeros({}, chx::Dtype::kFloat32);
            chx::Array acc = chx::Zeros({}, chx::Dtype::kFloat32);

            for (int64_t i = 0; i < n_test; i += batch_size) {
                std::vector<chx::ArrayIndex> indices{chx::Slice{i, i + batch_size}};
                chx::Array x = test_x.At(indices);
                chx::Array t = test_t.At(indices);
                chx::Array y = model(x);
                loss += SoftmaxCrossEntropy(y, t, false);
                acc += (y.ArgMax(1).AsType(t.dtype()) == t).Sum().AsType(acc.dtype());
            }

            std::cout << "epoch: " << epoch << " loss=" << chx::AsScalar(loss / n_test) << " accuracy=" << chx::AsScalar(acc / n_test)
                      << " elapsed_time=" << std::chrono::duration<double>{std::chrono::high_resolution_clock::now() - start}.count()
                      << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    int64_t epochs{20};
    int64_t batch_size{100};
    int64_t n_hidden{1000};
    int64_t n_layers{3};
    float lr{0.01};
    std::string device_name{"native"};
    std::string mnist_root{"./"};

    for (int i = 1; i < argc - 1; i += 2) {
        std::string arg = argv[i];
        if (arg == "--epoch") {
            epochs = std::atoi(argv[i + 1]);
        } else if (arg == "--batchsize") {
            batch_size = std::atoi(argv[i + 1]);
        } else if (arg == "--unit") {
            n_hidden = std::atoi(argv[i + 1]);
        } else if (arg == "--layer") {
            n_layers = std::atoi(argv[i + 1]);
        } else if (arg == "--lr") {
            lr = std::atof(argv[i + 1]);
        } else if (arg == "--device") {
            device_name = argv[i + 1];
        } else if (arg == "--data") {
            mnist_root = argv[i + 1];
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (mnist_root.empty()) {
        throw std::runtime_error("MNIST directory cannot be empty.");
    }
    if (mnist_root.back() != '/') {
        mnist_root += "/";
    }

    chx::Context ctx{};
    chx::SetDefaultContext(&ctx);
    chx::Device& device = ctx.GetDevice(device_name);
    chx::SetDefaultDevice(&device);

    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Minibatch size: " << batch_size << std::endl;
    std::cout << "Hidden neurons: " << n_hidden << std::endl;
    std::cout << "Layers: " << n_layers << std::endl;
    std::cout << "Learning rate: " << lr << std::endl;
    std::cout << "Device: " << device.name() << std::endl;
    std::cout << "MNIST root: " << mnist_root << std::endl;

    Run(epochs, batch_size, n_hidden, n_layers, lr, mnist_root);
}
