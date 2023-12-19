#pragma once
#include "../nn/layer.h"
#include <iostream>

using namespace mlx::core;
 
class Agent
{
public:
    Agent(int state_size, int action_size, std::function<array(const array&)> score_state);
    ~Agent();
    
    void trainOffline(array current_state, array previous_state, array current_action, array previous_action);
    array process(array current_state, bool debug = false);
protected:
    float train_state_predictor(array current_state, float learning_rate);
    float train_action_generator(array current_state, float learning_rate);
    array generate_action(array current_state, float aspiration_rate);
    array select_action(array current_state);
    void commit_memory(array current_state, array current_action);
    
    Model state_predictor;
    Model action_generator;
    
    array prev_state;
    array prev_action;
    
    std::function<array(const array&)> score_state_fn;
    
    std::vector<array> actions;
};
