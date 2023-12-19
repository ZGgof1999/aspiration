#include "Environment.h"

Environment::Environment() { rocket = std::make_unique<Rocket_StatePredictorActionGenerator>(); }
Environment::~Environment() { }

void Environment::render()
{
    ImGuiIO &io = ImGui::GetIO();
    
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    draw->AddCircleFilled(io.MousePos, 10, IM_COL32(205, 25, 25, 255));
    
    static bool ai;
    ImGui::Checkbox("AI", &ai);
    if (ai)
        rocket->computeControls(io.MousePos);
    else
    {
        static float main_thrust;
        ImGui::SliderFloat("Main-Thrust", &main_thrust, 0.0f, 1.0f);
        static float rot_thrust;
        ImGui::SliderFloat("Rot-Thrust", &rot_thrust, -1.0f, 1.0f);
        rocket->setControls(main_thrust, rot_thrust);
    }
    
    static bool physics;
    ImGui::Checkbox("Physics", &physics);
    if (physics)
        rocket->applyPhysics();
    
    static bool reset;
    ImGui::Checkbox("Reset", &reset);
    if (reset)
        rocket = std::make_unique<Rocket_StatePredictorActionGenerator>();
    reset = false;
    
    rocket->draw();
}
