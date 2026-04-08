#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <typeinfo>
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



};



int main() {
    Node a(1.0, {}, "");
    Node b(2.0, {}, "");
    Node c = a + b;
    Node d = a + 1;

    cout << "c.data = " << c.data << endl; 
    cout << "d.data = " << d.data << endl; 

    return 0;
}


