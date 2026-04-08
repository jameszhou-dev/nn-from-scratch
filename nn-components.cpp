#include "nn.h"
#include <random>
using namespace std;

random_device rd;
mt19937 gen(67);
uniform_real_distribution<float> dist(-1.0, 1.0);


Value::Value() {
    this->data = 0.0;
}

Value::Value(float data) {
    this->data = data;
}

Value::Value(float data, string label) {
    this->data = data;
    this->label = label;
}

shared_ptr<Value> Value::operator+(shared_ptr<Value> other) { // addition with another node
    auto out = make_shared<Value>(this->data + other->data);
    out->operation = "+";
    out->prev.push_back(shared_from_this());
    out->prev.push_back(other);
    out->_backward = [this, other, out]() mutable {
        this->grad += 1.0 * out->grad;
        other->grad += 1.0 * out->grad;
    };
    return out;
}

shared_ptr<Value> Value::operator+(float data) { // addition with float
    return operator+(make_shared<Value>(data));
}

shared_ptr<Value> Value::operator*(std::shared_ptr<Value> other) { // multiplication with another node
    auto out = make_shared<Value>(this->data * other->data);
    out->operation = "*";
    out->prev.push_back(shared_from_this());
    out->prev.push_back(other);
    out->_backward = [this, other, out]() mutable {
        this->grad += other->data * out->grad;
        other->grad += this->data * out->grad;
    };
    return out;
}

shared_ptr<Value> Value::operator*(float data) { // multiplication with a float
    return operator*(make_shared<Value>(data));
}

shared_ptr<Value> Value::pow(float data) { // power with float
    auto out = make_shared<Value>(powf(this->data, data));
    out->operation = "^";
    out->prev.push_back(shared_from_this());
    out->_backward = [this, out, data]() mutable {
        this->grad += data * powf(this->data, data - 1) * out->grad;
    };
    return out;
}

shared_ptr<Value> Value::operator/(std::shared_ptr<Value> other) { // division with another node
    return operator*(other->pow(-1));
}

shared_ptr<Value> Value::operator-(std::shared_ptr<Value> other) { // subtraction with another node
    return operator+(other->operator*(-1));
}

shared_ptr<Value> Value::operator-(float data) { // subtraction with another node
    return operator-(make_shared<Value>(data));
}

shared_ptr<Value> Value::tanh() {
    float x = this->data;
    float t = (expf(2*x) - 1) / (expf(2*x) + 1);
    auto out = make_shared<Value>(t);
    out->operation = "tanh";
    out->prev.push_back(shared_from_this());
    out->_backward = [this, out, t]() mutable {
        this->grad += (1 - powf(t, 2)) * out->grad;
    };
    return out;
}

shared_ptr<Value> Value::node_exp() {
    float x = this->data;
    auto out = make_shared<Value>(expf(x));
    out->operation = "exp";
    out->prev.push_back(shared_from_this());
    out->_backward = [this, out]() mutable {
        this->grad += out->data * out->grad;
    };
    return out;
}

void Value::backward() {
    vector<shared_ptr<Value>> topo;
    set<Value*> visited;
    function<void(shared_ptr<Value>)> build_topo = [&](shared_ptr<Value> node) {
        if (!visited.count(node.get())) {
            visited.insert(node.get());
            for (auto child : node->prev) {
                build_topo(child);
            }
            topo.push_back(node);
        }
    };
    build_topo(shared_from_this());
    this->grad = 1.0;
    for (int i = topo.size() - 1; i >= 0; i--) {
        topo[i]->_backward();
    }
}


ostream& operator<<(ostream& os, const Value& value) {
    os << "Value(data=" << value.data << ", grad=" << value.grad << ")";
    return os;
}



Neuron::Neuron(int num_inputs) {
    this->num_inputs = num_inputs;
    for (int i = 0; i < this->num_inputs; i++) {
        weights.push_back(Value::make(dist(gen)));
    }
    bias = Value::make(dist(gen));
}

shared_ptr<Value> Neuron::forward(vector<shared_ptr<Value>> inputs) {
    shared_ptr<Value> sum = bias;
    for (int i = 0; i < this->num_inputs; i++) {
        shared_ptr<Value> weighted_input = *weights[i] * inputs[i];
        sum = *sum + weighted_input;
    }
    return sum->tanh();
}

shared_ptr<Value> Neuron::forward(vector<float> inputs) {
    shared_ptr<Value> sum = bias;
    for (int i = 0; i < this->num_inputs; i++) {
        shared_ptr<Value> weighted_input = *weights[i] * inputs[i];
        sum = *sum + weighted_input;
    }
    return sum->tanh();
}



Layer::Layer(int num_inputs, int num_outputs) {
    this->num_inputs = num_inputs;
    this->num_outputs = num_outputs;
    for (int i = 0; i < num_outputs; i++) {
        neurons.push_back(Neuron(num_inputs));
    }
}

vector<shared_ptr<Value>> Layer::forward(vector<shared_ptr<Value>> inputs) {
    vector<shared_ptr<Value>> output;
    for (int i = 0; i < num_outputs; i++) {
        output.push_back(neurons[i].forward(inputs));
    }
    return output;
}

vector<shared_ptr<Value>> Layer::forward(vector<float> inputs) {
    vector<shared_ptr<Value>> output;
    for (int i = 0; i < num_outputs; i++) {
        output.push_back(neurons[i].forward(inputs));
    }
    return output;
}



MLP::MLP(int num_inputs, vector<int> num_outputs) {
    vector<int> sz;
    sz.push_back(num_inputs);
    sz.insert(sz.end(), num_outputs.begin(), num_outputs.end());
    for (int i = 0; i < num_outputs.size(); i++) {
        layers.push_back(Layer(sz[i], sz[i+1]));
    }
}

vector<shared_ptr<Value>> MLP::forward(vector<shared_ptr<Value>> inputs) {
    vector<shared_ptr<Value>> output;
    for (int i = 0; i < layers.size(); i++) {
        if (i == 0) {
            output = layers[i].forward(inputs); 
        } else {
            output = layers[i].forward(output);  
        }
    }
    return output;
}

vector<shared_ptr<Value>> MLP::forward(vector<float> inputs) {
    vector<shared_ptr<Value>> output;
    for (int i = 0; i < layers.size(); i++) {
        if (i == 0) {
            output = layers[i].forward(inputs); 
        } else {
            output = layers[i].forward(output); 
        }
    }
    return output;
}