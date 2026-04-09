#include "nn.h"
#include <fstream>  
#include <sstream>  
#include <vector>  
#include <string> 
#include <algorithm>
#include <random>
using namespace std;

struct iris_row{
    vector<float> features;
    int label;
};

int read_file(string& filename, vector<iris_row> &iris_dataset) {
    ifstream file(filename);
    string line;
    if (file.is_open()) {
        while (getline(file, line)) {
            stringstream ss(line);
            string temp;
            iris_row row;
            for (int i = 0; i < 4; i++) {
                getline(ss, temp, ',');
                row.features.push_back(stof(temp));
            }
            getline(ss, temp, ',');
            if (temp == "Iris-setosa") {
                row.label = 0;
            } else if (temp == "Iris-versicolor") {
                row.label = 1;
            } else if (temp == "Iris-virginica") {
                row.label = 2;
            }
            iris_dataset.push_back(row);
        }
        file.close();
        return 0;
    } else {
        return 1;
    }
}
vector<vector<shared_ptr<Value>>> forward_pass(MLP &mlp, vector<vector<float>> x) {
    vector<vector<shared_ptr<Value>>> y_pred;
    for (int i = 0; i < x.size(); i++) {
        vector<shared_ptr<Value>> output = mlp.forward(x[i]);
        y_pred.push_back(output);
    }
    return y_pred;
}

shared_ptr<Value> calc_loss(vector<vector<shared_ptr<Value>>> y_pred, vector<float> y) {
    shared_ptr<Value> loss = Value::make(0.0);
    vector<shared_ptr<Value>> neuron_store;
    for (int i = 0; i < y_pred.size(); i++) {
        int correct_class = (int)y[i];
        for (int j = 0; j < y_pred[i].size(); j++) {
            float target = (j == correct_class) ? 1.0f : 0.0f;
            shared_ptr<Value> ygt = Value::make(target);
            shared_ptr<Value> diff = *y_pred[i][j] - ygt;
            shared_ptr<Value> mse = diff->pow(2);
            neuron_store.push_back(ygt);
            neuron_store.push_back(diff);
            neuron_store.push_back(mse);
            shared_ptr<Value> new_loss = *loss + mse;
            neuron_store.push_back(loss);
            loss = new_loss;
        }
    }
    float n = (float)(y_pred.size() * y_pred[0].size());
    shared_ptr<Value> mean = Value::make(1.0f / n);
    loss = *loss * mean;
    return loss;
}


int main() {
    vector <iris_row> iris_dataset;
    string path = "data/iris.csv";
    if (read_file(path, iris_dataset) == 1) {
        cout << "error reading file" << endl;
        return 1;
    }

    MLP mlp(4, {16, 8, 3}); // 4 inputs, 2 hidden layers (16 neurons, 8 neurons), 3 output
    vector<vector<float>> x;
    vector<float> y;
    for (int i = 0; i < iris_dataset.size(); i++) {
        x.push_back(iris_dataset[i].features);
        y.push_back(iris_dataset[i].label);
    }
    int num_epochs = 100;
    double learning_rate = 0.001;
    int batch_size = 16;
    for (int i = 1; i < num_epochs + 1; i++) {
        cout << "epoch: " << i;
        shuffle(iris_dataset.begin(), iris_dataset.end(), default_random_engine(i)); // shuffle dataset each time
        int batch_size = 16; // num batches
        shared_ptr<Value> loss;
        for (int i = 0; i < x.size(); i += batch_size) {
            int end = min((int)x.size(), i + batch_size);
            vector<vector<float>> x_batch(x.begin() + i, x.begin() + end);
            vector<float> y_batch(y.begin() + i, y.begin() + end);

            // forward pass
            vector<vector<shared_ptr<Value>>> y_pred = forward_pass(mlp, x_batch);
            // zero grad
            vector<shared_ptr<Value>> params = mlp.parameters();
            for (int i = 0; i < params.size(); i++) {   
                params[i]->grad = 0;
            }
            // calculate loss
            loss = calc_loss(y_pred, y_batch);
            // backward pass
            loss->backward();
            learning_rate = 0.001 * (1.0 / (1.0 + 0.01 * i)); // decrease learning rate each epoch
            for (int i = 0; i < params.size(); i++) { 
                params[i]->data = params[i]->data + (-(learning_rate) * params[i]->grad); 
            }
        }
        cout << " loss: " << loss->data << endl;
    }
    return 0;
}