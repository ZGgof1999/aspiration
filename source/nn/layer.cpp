#include "layer.h"

Layer::Layer (int input_size, int output_size, bool activation_fn) :
    m_input(zeros({1, input_size})),
    m_weights(random::normal({input_size, output_size})),
    m_biases(random::normal({1, output_size})),
    m_output(zeros({1, output_size})),
    m_activation_fn([](const array& input) { return sigmoid(input); }),
    m_activation_grad_fn([](const array& input) { return sigmoid(input) * (ones(input.shape()) - sigmoid(input)); })
{ 
    if (!activation_fn)
    {
        m_activation_fn = [](const array& input) { return input; };
        m_activation_grad_fn = [](const array& input) { return ones(input.shape()); };
    }
}

array Layer::forward(const array& input)
{
    m_input = input;
    m_output = m_activation_fn(matmul(input, m_weights) + m_biases);
    return m_output;
}

array Layer::backward(const array& output_grad, float learning_rate) {
    // Compute gradients for weights and biases
    auto activation_derivative = m_activation_grad_fn(m_output);
    auto activation_grad = activation_derivative * output_grad;
    auto input_grad = matmul(activation_grad, transpose(m_weights));
    auto weights_grad = transpose(matmul(transpose(activation_grad), m_input));
    auto biases_grad = sum(activation_grad, 0);
    
    // Update weights and biases
    m_weights = m_weights - learning_rate * weights_grad;
    m_biases = m_biases - learning_rate * biases_grad;

    return input_grad;
}


Model::Model() { }
Model::~Model() { }
    
void Model::addLayer(Layer new_layer) { layers.push_back(new_layer); }
array Model::forward(array input) 
{
    array output = input;
    for (Layer& layer : layers)
        output = layer.forward(output);
    return output;
}
void Model::backward(array gradient, float learning_rate) 
{
    for (int l = static_cast<int>(layers.size()) - 1; l >= 0; l--)
        gradient = layers[l].backward(gradient, learning_rate);
}
