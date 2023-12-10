#include "Environment.h"

Environment::Environment() { rocket = std::make_unique<Rocket>(); }
Environment::~Environment() { }

void Environment::render()
{
    ImGuiIO &io = ImGui::GetIO();
    
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    draw->AddCircleFilled(io.MousePos, 10, IM_COL32(205, 25, 25, 255));
    
    static float thrust;
    ImGui::SliderFloat("Thrust", &thrust, 0.0f, 1.0f);
    
    static float rot_thrust;
    ImGui::SliderFloat("Rot-Thrust", &rot_thrust, -1.0f, 1.0f);
    
    static bool physics;
    ImGui::Checkbox("Physics", &physics);
    if (physics)
        rocket->applyPhysics(thrust, rot_thrust);
    
    static bool reset;
    ImGui::Checkbox("Reset", &reset);
    if (reset)
        rocket = std::make_unique<Rocket>();
    reset = false;
    
    rocket->draw();
}
