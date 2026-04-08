#ifndef NN_H
#define NN_H
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <set>
#include <cmath>

class Value {
public:
    float data;
    float grad = 0.0;
    std::vector<Value*> prev;
    std::string label = "";
    std::string operation = "";
    std::function<void()> _backward = [](){};

    Value(float data, std::vector<Value*> children, std::string operation);
    Value(float data);
    Value(float data, std::string label);
    Value operator+(Value& other);
    Value operator+(float data);
    Value operator*(Value& other);
    Value operator*(float data);
    Value pow(float other);
    Value operator/(Value& other);
    Value operator-(Value& other);
    Value tanh();
    Value node_exp();
    void backward();

    friend std::ostream& operator<<(std::ostream& os, const Value& value);
};

#endif