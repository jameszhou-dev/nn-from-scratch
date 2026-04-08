#include "nn.h"
using namespace std;

int main() {
    // test value
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
    /*cout << "o.grad: " << o.grad << endl;
    cout << "x1.grad: " << x1.grad << endl;
    cout << "x2.grad: " << x2.grad << endl;
    cout << "w1.grad: " << w1.grad << endl;
    cout << "w2.grad: " << w2.grad << endl;
    cout << "x1w1.grad: " << x1w1.grad << endl;
    cout << "x2w2.grad: " << x2w2.grad << endl;*/

    // test neuron
    vector<float> x = {2.0, 3.0, -1.0};
    MLP mlp(3, {4, 4, 1});
    vector<Value> output = mlp.forward(x);
    for (int i = 0; i < output.size(); i++) {
        cout << output[i] << endl;
    }

}