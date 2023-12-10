#pragma once
#include "../nn/layer.h"
#include "../../imgui/imgui.h"

class Rocket
{
public:
    Rocket();
    ~Rocket();
    
    void applyPhysics(float t, float rt);
    void draw();
private:
    array position = array({200.0f, 0.0f}, {1, 2});
    array velocity = array({0.0f, 0.0f}, {1, 2});
    array acceleration = array({0.0f, 0.0f}, {1, 2});
    
    float rotation { 0.0f };
    float rotation_dt { 0.0f };
    float rotation_dt_dt { 0.0f };
    
    const float GRAV_MAG { 9.8f };
    const array gravity = array({
        0.0f, -GRAV_MAG}, {1, 2});
    
    const float MAIN_THRUST_MAG { 12.5f };
    const array main_thrust = array({0.0f, MAIN_THRUST_MAG}, {1, 2});
    float main_thrust_amount { 0.0f };
    
    const float ROT_THRUST_MAG { 3.0f };
    float rot_thrust_amount { 0.0f };
};
