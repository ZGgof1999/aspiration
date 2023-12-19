#include "Environment.h"

Environment::Environment() { rocket = std::make_unique<Rocket_StatePredictorActionGenerator>(); }
Environment::~Environment() { }

void Environment::render()
{
    ImGuiIO &io = ImGui::GetIO();
    
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    draw->AddCircleFilled(io.MousePos, 10, IM_COL32(205, 25, 25, 255));
    
    static bool pause;
    ImGui::Checkbox("Pause", &pause);
    if (pause)
        return;
    
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
        
        bool w = io.KeysDown[ImGuiKey_W];
        bool s = io.KeysDown[ImGuiKey_S];
        bool a = io.KeysDown[ImGuiKey_A];
        bool d = io.KeysDown[ImGuiKey_D];
        if (w)
            main_thrust = std::clamp(main_thrust + 0.05f, 0.0f, 1.0f);
        if (s)
            main_thrust = std::clamp(main_thrust - 0.05f, 0.0f, 1.0f);
        if (a)
            rot_thrust = std::clamp(rot_thrust + 0.05f, -1.0f, 1.0f);
        if (d)
            rot_thrust = std::clamp(rot_thrust - 0.05f, -1.0f, 1.0f);
        rocket->setControls(main_thrust, rot_thrust);
        rocket->humanControlTraining(io.MousePos);
    }
    rocket->applyPhysics();
    
    static bool reset;
    ImGui::Checkbox("Reset", &reset);
    if (reset)
        rocket = std::make_unique<Rocket_StatePredictorActionGenerator>();
    reset = false;
    
    rocket->draw();
}
