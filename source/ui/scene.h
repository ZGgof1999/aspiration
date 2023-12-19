#pragma once
#include <atomic>
#include <iostream>
#include <random>
#include "../DotPathFinder/grid.h"
#include "../Rockets/Environment.h"
class Scene
{
public:
    Scene();
    ~Scene();
    
    void render();
private:
    Grid grid;
};
