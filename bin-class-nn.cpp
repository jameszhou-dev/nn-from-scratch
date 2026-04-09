#include "nn.h"
using namespace std;

vector<shared_ptr<Value>> forward_pass(MLP &mlp, vector<vector<float>> xs) {
    vector<shared_ptr<Value>> ypred;
    cout << "predicted vals: ";
    for (int i = 0; i < xs.size(); i++) {
        vector<shared_ptr<Value>> output = mlp.forward(xs[i]);
        ypred.push_back(output[0]);
        cout << ypred[i]->data << " ";
    }
    cout << endl;
    return ypred;
}

shared_ptr<Value> calc_loss(vector<shared_ptr<Value>> ypred, float ys[]) {
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
    return loss;
}
int main() {
    MLP mlp(3, {4, 4, 1});
    vector<vector<float>> xs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}};
    float ys[] = {1.0, -1.0, -1.0, 1.0};


    int epoch = 0;
    for (int i = 0; i < 17; i++) {
        cout << "epoch: " << epoch << " " << endl;
        vector<shared_ptr<Value>> ypred = forward_pass(mlp, xs);
        shared_ptr<Value> loss = calc_loss(ypred, ys);
        loss->backward();
        vector<shared_ptr<Value>> params = mlp.parameters();
        for (int i = 0; i < params.size(); i++) { 
            params[i]->data = params[i]->data + (-0.1 * params[i]->grad); // change data based on gradient
        }
        cout << "loss: " << *loss << endl;
        cout << "___________________________________" << endl;
        epoch+=1;
    }
    return 0;
}