#pragma once
#include "imgui.h"
#include "mlx/mlx.h"
#include <atomic>
#include <iostream>

using namespace mlx::core;
 
class Layer
{
public:
    Layer(int input_size, int output_size);
    array forward(const array& input);
    array backward(const array& output_grad, float learning_rate);
private:
    array m_input;
    array m_weights;
    array m_biases;
    array m_output;
    std::function<array(const array&)> m_activation_fn;
    std::function<array(const array&)> m_activation_grad_fn;
};
