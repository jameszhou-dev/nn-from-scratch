#ifndef NN_H
#define NN_H

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <set>
#include <cmath>
#include <memory>

class Value : public std::enable_shared_from_this<Value> {  
public:
    float data;
    float grad = 0.0;
    std::vector<std::shared_ptr<Value>> prev;
    std::string label = "";
    std::string operation = "";
    std::function<void()> _backward = [](){};

    Value();
    Value(float data);
    Value(float data, std::string label);

    static std::shared_ptr<Value> make(float data, std::string label = "") {
        return std::make_shared<Value>(data, label);
    }

    std::shared_ptr<Value> operator+(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator+(float data);
    std::shared_ptr<Value> operator*(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator*(float data);
    std::shared_ptr<Value> pow(float data);
    std::shared_ptr<Value> operator/(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator-(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator-(float data);
    std::shared_ptr<Value> tanh();
    std::shared_ptr<Value> node_exp();
    void backward();

    friend std::ostream& operator<<(std::ostream& os, const Value& value);
};

class Neuron {
public:
    int num_inputs;
    std::vector<std::shared_ptr<Value>> weights;
    std::shared_ptr<Value> bias;

    Neuron(int num_inputs);
    std::shared_ptr<Value> forward(std::vector<std::shared_ptr<Value>> inputs);
    std::shared_ptr<Value> forward(std::vector<float> inputs);
};

class Layer {
public:
    int num_inputs;
    int num_outputs;
    std::vector<Neuron> neurons;

    Layer(int num_inputs, int num_outputs);
    std::vector<std::shared_ptr<Value>> forward(std::vector<std::shared_ptr<Value>> inputs);
    std::vector<std::shared_ptr<Value>> forward(std::vector<float> inputs);
};

class MLP {
public:
    std::vector<Layer> layers;

    MLP(int num_inputs, std::vector<int> num_outputs);
    std::vector<std::shared_ptr<Value>> forward(std::vector<std::shared_ptr<Value>> inputs);
    std::vector<std::shared_ptr<Value>> forward(std::vector<float> inputs);
};

#endif