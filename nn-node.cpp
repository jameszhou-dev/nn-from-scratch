#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <typeinfo>
#include <cmath>
#include <set>
using namespace std;

class Value {
    public:
        float data;
        float grad = 0.0;
        vector<Value*> prev;
        string label = "";
        string operation = "";
        function<void()> _backward = [](){};
    
    Value(float data) {
        this->data = data;
    }
    Value(float data, string label) {
        this->data = data;
        this->label = label;

    }
    Value(float data, vector<Value*> children, string operation) {
        this->data = data;
        prev = children;
        this->operation = operation;
    }

    Value operator+(Value& other) { // addition with another node
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
    Value operator+(float data) { // addition with a float
        Value other(data);
        return operator+(other);
    }

    Value operator*(Value& other) { // multiplication with another node
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

    Value operator*(float data) { // multiplication with a float
        Value other(data);
        return operator*(other);
    }

    Value pow(float other) { // power with float
        vector<Value*> children;
        children.push_back(this);
        Value out(powf(this->data, other), children, "^");
        out._backward = [this, &other, &out]() {
            this->grad += other * (powf(this->data, other-1)) * out.grad;
        };

        return out;
    }

    Value operator/(Value& other) { // division with another node
        Value temp = other.pow(-1);
        return (*this) * temp;
    }

    Value operator-(Value& other) { // subtraction with another node
        Value temp = other * (-1);
        return (*this) + temp;
    }

    Value tanh() {
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

    Value node_exp() {
        float x = this->data;
        vector<Value*> children;
        children.push_back(this);
        Value out(exp(x), children, "exp");
        out._backward = [this, &out]() {
            this->grad += out.data * out.grad;
        };
        return out;
    }

    void backward() {
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

};



int main() {
    Value x1(2.0, "x1"); // input 1
    Value x2(0.0, "x2"); // input 2
    Value w1(-3.0, "w1"); // weight 1
    Value w2(1.0, "w2"); // weight 2
    Value b(6.8813735870195432, "b"); // bias
    // calculate weighted inputs
    Value x1w1 = x1*w1;
    x1w1.label = "x1*w1";
    Value x2w2 = x2*w2;
    x2w2.label = "x2*w2";
    Value x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2.label = "x1*w1 + x2*w2";
    // add bias to weighted inputs
    Value n = x1w1x2w2 + b; 
    n.label = "n";
    // apply activation function onto n
    Value o = n.tanh();
    o.label = "o";
    // back prop on o
    o.backward();
    cout << "o.grad: " << o.grad << endl;
    cout << "x1.grad: " << x1.grad << endl;
    cout << "x2.grad: " << x2.grad << endl;
    cout << "w1.grad: " << w1.grad << endl;
    cout << "w2.grad: " << w2.grad << endl;
    cout << "x1w1.grad: " << x1w1.grad << endl;
    cout << "x2w2.grad: " << x2w2.grad << endl;
    return 0;
}