#include "nn.h"
using namespace std;

int main() {
    MLP mlp(3, {4, 4, 1});
    vector<vector<float>> xs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}};
    float ys[] = {1.0, -1.0, -1.0, 1.0};

    vector<shared_ptr<Value>> ypred;
    for (int i = 0; i < xs.size(); i++) {
        vector<shared_ptr<Value>> output = mlp.forward(xs[i]);
        ypred.push_back(output[0]);
        cout << "prediction: " << *ypred[i] << endl;
    }
    vector<shared_ptr<Value>> neuron_store;

    shared_ptr<Value> loss = Value::make(0.0); 
    for (int i = 0; i < ypred.size(); i++) {
        shared_ptr<Value> ygt = Value::make(ys[i]);
        shared_ptr<Value> diff = *ypred[i] - ygt;
        shared_ptr<Value> mse = diff->pow(2);

        neuron_store.push_back(ygt);
        neuron_store.push_back(diff);
        neuron_store.push_back(mse);

        shared_ptr<Value> new_loss = *loss + mse;
        neuron_store.push_back(loss); 
        loss = new_loss;
    }

    cout << "loss: " << *loss << endl;
    loss->backward();
    cout << "loss after: " << *loss << endl;
    return 0;
}