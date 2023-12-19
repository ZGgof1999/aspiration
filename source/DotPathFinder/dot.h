#pragma once
#include <atomic>
#include <iostream>
#include "../nn/layer.h"

using namespace mlx::core;

enum Action
{
    stay = 0,
    left,
    right,
    up,
    down
};

struct TrainingData
{
    std::vector<float> x;
    std::vector<float> y;
};

struct Target
{
    int x;
    int y;
};

class Dot
{
public:
    Dot();
    ~Dot();
    
    void train(int num_batches, float learning_rate);
    void act(float target_x, float target_y);
    
    float x { 0 };
    float y { 0 };
private:
    Model model;
    
    std::vector<TrainingData> trainingData;
};

class OnlineDot
{
public:
    OnlineDot();
    ~OnlineDot();

    void act(float target_x, float target_y, float state_rate, float action_rate);
    
    float x { 0 };
    float y { 0 };
private:
    float getStateScore(array state);
    Model state_predictor;
    Model action_generator;
    
    array prev_state;
    array prev_action;
    
    std::vector<array> actions;
};
