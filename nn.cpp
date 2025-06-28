#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>





class Layer{
public:
  Eigen::MatrixXf outputs;
  Eigen::MatrixXf biases;
  Eigen::MatrixXf weights;

  Layer(){}
  
  Layer(int neuron_count) {
    outputs = (Eigen::MatrixXf::Zero(neuron_count, 1));
    biases = (Eigen::MatrixXf::Zero(neuron_count,1 ));
  }
  Layer next_layer(int neuron_count);
  };

Eigen::MatrixXf sigmoid_matrix(const Eigen::MatrixXf& x){
  return 1.0 / (1.0 + (-x.array()).exp());
}
void forward(Layer& curr, Layer& prev){
  curr.outputs = sigmoid_matrix((prev.weights.transpose() * prev.outputs) + curr.biases);
}

Layer Layer::next_layer(int neuron_count){

  Layer next = Layer(neuron_count);
  this->weights = Eigen::MatrixXf::Random(this->outputs.rows(), next.outputs.rows());
  return next;
}




class NN{
public:
  Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
  void backpropagation(const Eigen::MatrixXf& expected);
  void insert_layer(Layer& layer);
private:
  float learning_rate = 0.01;
  std::vector<Layer> layers;
};

void NN::insert_layer(Layer& layer){
  layers.push_back(layer);
}

Eigen::MatrixXf NN::forward(const Eigen::MatrixXf& input){

  layers[0].outputs = input;
  for(size_t i = 1; i < layers.size(); i++){

    Layer& curr = layers[i];
    Layer& prev = layers[i - 1];
    ::forward(curr, prev);
  }
  return layers[layers.size() - 1].outputs;
}


float error(const Eigen::MatrixXf& out, const Eigen::MatrixXf& exp){
  return ((out-exp).array().square().sum())/out.cols();
}

void NN::backpropagation(const Eigen::MatrixXf& expected){
  Eigen::MatrixXf& output = layers[layers.size() - 1].outputs;
  Eigen::MatrixXf delta = (output - expected).array() * output.array() * (1.0f - output.array());


  for (size_t i = layers.size() - 1; i > 0; i--){
    Layer& curr = layers[i];
    Layer& prev = layers[i - 1];

    curr.biases += delta * (-learning_rate);

    prev.weights += (prev.outputs * delta.transpose()) * (-learning_rate);
  
    Eigen::MatrixXf sigmoid_derivate = prev.outputs.array() * (1.0f - prev.outputs.array());

    delta = (prev.weights * delta).array() * sigmoid_derivate.array();
  }
}


class Dataset {
public:
  virtual int count() const = 0;
  virtual Eigen::MatrixXf get_input(int index) const = 0;
  virtual Eigen::MatrixXf get_output(int index) const = 0;
};


void train(NN& nn, const Dataset& dataset, int epoch) {
  for (int _ = 0; _ < epoch; _++) {
    for (int i = 0; i < dataset.count(); i++) {
      Eigen::MatrixXf input    = dataset.get_input(i);
      Eigen::MatrixXf expected = dataset.get_output(i);

      Eigen::MatrixXf outputs = nn.forward(input);
      nn.backpropagation(expected);

      printf("Error: %.4f\n", error(outputs, expected));
    }
  }
}
class DsMinist : public Dataset {
public:
  DsMinist(const std::string& labels_path, const std::string& images_path) {
    load_labels(labels_path);
    load_images(images_path);
    if (images.size() != labels.size()) {
      throw std::runtime_error("Mismatch between image and label count");
    }
  }

  int count() const override {
    return images.size();
  }

  Eigen::MatrixXf get_input(int index) const override {
    return images[index];
  }

  Eigen::MatrixXf get_output(int index) const override {
    return labels[index];
  }

private:
  std::vector<Eigen::MatrixXf> images;
  std::vector<Eigen::MatrixXf> labels;

  void load_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open label file");

    int32_t magic_number = 0, num_items = 0;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_items), 4);
    magic_number = __builtin_bswap32(magic_number);
    num_items    = __builtin_bswap32(num_items);

    labels.resize(num_items);
    for (int i = 0; i < num_items; ++i) {
      unsigned char label;
      file.read(reinterpret_cast<char*>(&label), 1);
      Eigen::MatrixXf one_hot = Eigen::MatrixXf::Zero(10, 1);
      one_hot(static_cast<int>(label), 0) = 1.0f;
      labels[i] = one_hot;
    }
  }

  void load_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open image file");

    int32_t magic_number = 0, num_items = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_items), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);
    magic_number = __builtin_bswap32(magic_number);
    num_items    = __builtin_bswap32(num_items);
    rows         = __builtin_bswap32(rows);
    cols         = __builtin_bswap32(cols);

    images.resize(num_items);
    for (int i = 0; i < num_items; ++i) {
      Eigen::MatrixXf img(rows * cols, 1);
      for (int j = 0; j < rows * cols; ++j) {
        unsigned char pixel;
        file.read(reinterpret_cast<char*>(&pixel), 1);
        img(j, 0) = static_cast<float>(pixel) / 255.0f;  // Normalize to [0,1]
      }
      images[i] = img;
    }
  }
};

int get_argmax(const Eigen::MatrixXf& m) {
    Eigen::MatrixXf::Index row, col;
    m.maxCoeff(&row, &col);
    return row;
}

void test(NN& nn, const Dataset& dataset) {
  int correct_predictions = 0;
  for (int i = 0; i < 10; i++) {
    Eigen::MatrixXf input    = dataset.get_input(i);
    Eigen::MatrixXf expected = dataset.get_output(i);

    Eigen::MatrixXf outputs = nn.forward(input);

    int predicted = get_argmax(outputs);
    int expected_val = get_argmax(expected);
    if(predicted == expected_val) correct_predictions++;
    printf("Predicted: %d, Expected: %d\n", predicted, expected_val);
  }
  printf("Accuracy on first 10 samples: %.2f%%\n", (float)correct_predictions / 10.0f * 100.0f);
}
int main() {

  // Load training dataset.
  DsMinist dataset_train(
    "train-labels.idx1-ubyte",
    "train-images.idx3-ubyte");

  // Create layers.
  Layer input(784);
  Layer hidden_1 = input.next_layer(16);
  Layer hidden_2 = hidden_1.next_layer(10);
  Layer output   = hidden_2.next_layer(10);

  // Construct neural network.
  NN nn;
  nn.insert_layer(input);
  nn.insert_layer(hidden_1);
  nn.insert_layer(hidden_2);
  nn.insert_layer(output);

  // Train the model.
  train(nn, dataset_train, /*epoch=*/10);

  test(nn, dataset_train);

  return 0;
}
