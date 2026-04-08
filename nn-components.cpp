#include "nn.h"
#include <random>
using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dist(-1.0, 1.0);


Value::Value(float data, vector<Value*> children, string operation){
    this->data = data;
    prev = children;
    this->operation = operation;
}

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

Value Value::operator+(Value& other) { // addition with another node
    vector<Value*> children;
    children.push_back(this);
    children.push_back(&other);
    Value out(this->data + other.data, children, "+");

    out._backward = [this, &other, &out]() {
        this->grad += 1.0 * out.grad;
        other.grad += 1.0 * out.grad;
    };

    return out;
}

Value Value::operator+(float data) { // addition with float
    Value other(data);
    return operator+(other);
}

Value Value::operator*(Value& other) { // multiplication with another node
    vector<Value*> children;
    children.push_back(this);
    children.push_back(&other);
    Value out(this->data * other.data, children, "*");

    out._backward = [this, &other, &out]() {
        this->grad += other.data * out.grad;
        other.grad += this->data * out.grad;
    };

    return out;
}

Value Value::operator*(float data) { // multiplication with a float
    Value other(data);
    return operator*(other);
}

Value Value::pow(float other) { // power with float
    vector<Value*> children;
    children.push_back(this);
    Value out(powf(this->data, other), children, "^");
    out._backward = [this, &other, &out]() {
        this->grad += other * (powf(this->data, other-1)) * out.grad;
    };

    return out;
}

Value Value::operator/(Value& other) { // division with another node
    Value temp = other.pow(-1);
    return (*this) * temp;
}

Value Value::operator-(Value& other) { // subtraction with another node
    Value temp = other * (-1);
    return (*this) + temp;
}

Value Value::tanh() {
    float x = this->data;
    float t = (exp(2*x) - 1) / (exp(2*x) + 1);
    vector<Value*> children;
    children.push_back(this);
    Value out(t, children, "tanh");
    out._backward = [this, &out, t]() {
        this->grad += (1-powf(t, 2)) * out.grad;
    };
    return out;
}

Value Value::node_exp() {
    float x = this->data;
    vector<Value*> children;
    children.push_back(this);
    Value out(exp(x), children, "exp");
    out._backward = [this, &out]() {
        this->grad += out.data * out.grad;
    };
    return out;
}

void Value::backward() {
    vector<Value*> topo;
    set<Value*> visited;
    function<void(Value*)> build_topo = [&](Value* node) {
        if (!visited.count(node)) {
            visited.insert(node);  
            for (Value* child : node->prev) {
                build_topo(child);
            }
            topo.push_back(node);
        }
    };
    build_topo(this);
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
        weights.push_back(Value(dist(gen)));
    }
    bias = Value(dist(gen));
}

Value Neuron::forward(vector<Value> inputs) {
    Value sum = bias;
    for (int i = 0; i < this->num_inputs; i++) {
        Value weighted_input = weights[i] * inputs[i];
        sum = sum + weighted_input;
    }
    return sum.tanh();
}

Value Neuron::forward(vector<float> inputs) {
    Value sum = bias;
    for (int i = 0; i < this->num_inputs; i++) {
        Value weighted_input = weights[i] * inputs[i];
        sum = sum + weighted_input;
    }
    return sum.tanh();
}



Layer::Layer(int num_inputs, int num_outputs) {
    this->num_inputs = num_inputs;
    this->num_outputs = num_outputs;
    for (int i = 0; i < num_outputs; i++) {
        neurons.push_back(Neuron(num_inputs));
    }
}

vector<Value> Layer::forward(std::vector<Value> inputs){
    vector<Value> output;
    for (int i = 0; i < num_outputs; i++) {
        output.push_back(neurons[i].forward(inputs));
    }
    return output;
}

vector<Value> Layer::forward(std::vector<float> inputs){
    vector<Value> output;
    for (int i = 0; i < num_outputs; i++) {
        output.push_back(neurons[i].forward(inputs));
    }
    return output;
}




MLP::MLP(int num_inputs, vector<int> num_outputs) {
    vector<int> sz;
    sz.push_back(num_inputs);
    sz.insert(sz.end(), num_outputs.begin(), num_outputs.end());
}