#pragma once
#include <atomic>
#include <iostream>
#include <random>
#include "imgui/imgui.h"
#include "dot.h"

class Grid
{
public:
    Grid();
    ~Grid();
    
    void render();
private:
    ImVec2 getTopLeft(float x, float y, ImVec2 center);
    ImVec2 getBottomRight(float x, float y, ImVec2 center);
    
    Dot dot;
    float targetX { 0 };
    float targetY { 0 };
    
    float gridSize = 36;
    float halfGrid = 0.5 * gridSize;
    float padding = 4;

    int numBatches = 10;
    float learningRate = 0.02f;
};
