#pragma once
#include "../nn/layer.h"
#include "imgui/imgui.h"

class Rocket
{
public:
    Rocket();
    ~Rocket();
    
    virtual void computeControls(ImVec2 mouse_pos);
    void setControls(float thrust, float rot_thrust);
    void applyPhysics();
    void draw();
protected:
    array position = array({0.0f, 0.0f}, {1, 2});
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
    
    const float ROT_THRUST_MAG { 6.0f };
    float rot_thrust_amount { 0.0f };
};

class Rocket_HardCodedOptimalFunctionApproximation : public Rocket
{
public:
    Rocket_HardCodedOptimalFunctionApproximation();
    void computeControls(ImVec2 mouse_pos) override;
private:
    Model model;
};

class Rocket_StatePredictorActionGenerator : public Rocket
{
public:
    Rocket_StatePredictorActionGenerator();
    void computeControls(ImVec2 mouse_pos) override;
private:
    float getStateScore(array state);
    Model state_predictor;
    Model action_generator;
    
    array prev_state;
    array prev_action;
    
    std::vector<array> actions;
};
