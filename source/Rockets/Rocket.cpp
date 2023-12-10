#include "Rocket.h"

Rocket::Rocket() { }
Rocket::~Rocket() { }

void Rocket::applyPhysics(float t, float rt)
{
    float scale = 0.01;
    float friction = 0.99f;
    
    rot_thrust_amount = rt;
    rotation_dt_dt = scale * 0.01 * rot_thrust_amount;
    rotation_dt = friction * rotation_dt + rotation_dt_dt;
    rotation = rotation + rotation_dt;
    
    array rotation_matrix ({cos(rotation), sin(rotation),
                            -sin(rotation), cos(rotation)}, {2, 2});
    
    main_thrust_amount = t;
    acceleration = scale * (gravity + main_thrust_amount * matmul(main_thrust, rotation_matrix));
    velocity = friction * velocity + acceleration;
    position = position + velocity;
    position.eval();
}

void Rocket::draw()
{
    ImVec2 size = ImGui::GetWindowSize();
    
    float* data = position.data<float>();
    float x = data[0];
    float y = size.y - data[1];
    float theta = rotation;
    float r = 12.0f;
    
    float pi = M_PI;
    float PIo3 = 2 * pi * 0.4f;
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    draw->AddTriangleFilled({x - (0.8f + main_thrust_amount) * r * sin(theta + pi), y - (0.8f + main_thrust_amount) * r * cos(theta + pi)},
                            {x - r * sin(theta - PIo3), y - r * cos(theta - PIo3)},
                            {x - r * sin(theta + PIo3), y - r * cos(theta + PIo3)}, IM_COL32(245, 80, 15, 255));
    
    draw->AddTriangleFilled({x - r * sin(theta), y - r * cos(theta)},
                            {x - r * sin(theta + PIo3), y - r * cos(theta + PIo3)},
                            {x - r * sin(theta - PIo3), y - r * cos(theta - PIo3)}, IM_COL32(255, 255, 255, 255));
}
