#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <typeinfo>
#include <cmath>
#include <set>
using namespace std;

class Node {
    public:
        float data;
        float grad = 0.0;
        vector<Node*> prev;
        string label = "";
        string operation = "";
        function<void()> backward = [](){};
    
    Node(float data) {
        this->data = data;
    }
    Node(float data, vector<Node*> children, string operation) {
        this->data = data;
        prev = children;
        this->operation = operation;
    }

    Node operator+(Node& other) { // addition with another node
        vector<Node*> children;
        children.push_back(this);
        children.push_back(&other);
        Node out(this->data + other.data, children, "+");

        out.backward = [this, &other, &out]() {
            this->grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };

        return out;
    }
    Node operator+(float data) { // addition with a float
        Node other(data);
        return operator+(other);
    }

    Node operator*(Node& other) { // multiplication with another node
        vector<Node*> children;
        children.push_back(this);
        children.push_back(&other);
        Node out(this->data * other.data, children, "*");

        out.backward = [this, &other, &out]() {
            this->grad += other.data * out.grad;
            other.grad += this->data * out.grad;
        };

        return out;
    }

    Node operator*(float data) { // multiplication with a float
        Node other(data);
        return operator*(other);
    }

    Node pow(float other) { // power with float
        vector<Node*> children;
        children.push_back(this);
        Node out(powf(this->data, other), children, "^");
        out.backward = [this, &other, &out]() {
            this->grad += other * (powf(this->data, other-1)) * out.grad;
        };

        return out;
    }

    Node operator/(Node& other) { // division with another node
        Node temp = other.pow(-1);
        return (*this) * temp;
    }

    Node operator-(Node& other) { // subtraction with another node
        Node temp = other * (-1);
        return (*this) + temp;
    }

    Node tanh() {
        float x = this->data;
        float t = (exp(2*x) - 1) / (exp(2*x) + 1);
        vector<Node*> children;
        children.push_back(this);
        Node out(t, children, "tanh");
        out.backward = [this, &out, t]() {
            this->grad += (1-powf(t, 2)) * out.grad;
        };
        return out;
    }

    Node node_exp() {
        float x = this->data;
        vector<Node*> children;
        children.push_back(this);
        Node out(exp(x), children, "exp");
        out.backward = [this, &out]() {
            this->grad += out.data * out.grad;
        };
        return out;
    }

    void backward() {
        vector<Node*> topo;
        set<Node*> visited;
        function<void(Node*)> build_topo = [&](Node* node) {
            if (!visited.count(node)) {
                visited.insert(node);  
                for (Node* child : node->prev) {
                    build_topo(child);
                }
                topo.push_back(node);
            }
        };
        build_topo(this);
        this->grad = 1.0;
        for (int i = topo.size() - 1; i >= 0; i--) {
            topo[i]->backward();
        }
    }

};



int main() {
    Node a(3.0, {}, "");
    Node b(2.0, {}, "");
    Node c = a * b;
    Node d = a * 2;

    cout << "c.data = " << c.data << endl; 
    cout << "d.data = " << d.data << endl; 



    return 0;
}


