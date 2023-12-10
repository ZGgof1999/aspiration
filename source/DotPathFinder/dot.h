#pragma once
#include "mlx/mlx.h"
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
    Layer l1;
    Layer l2;
    Layer l3;
    
    std::vector<TrainingData> trainingData;
};
