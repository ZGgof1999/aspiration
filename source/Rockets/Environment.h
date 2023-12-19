#pragma once
#include "../../imgui/imgui.h"
#include "Rocket.h"

class Environment
{
public:
    Environment();
    ~Environment();
    
    void render();
private:
    std::unique_ptr<Rocket_StatePredictorActionGenerator> rocket;
};
