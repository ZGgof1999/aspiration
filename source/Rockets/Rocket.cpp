#include "Rocket.h"

Rocket::Rocket() { }
Rocket::~Rocket() { }

void Rocket::computeControls(ImVec2 mouse_pos) { }

void Rocket::setControls(float main_thrust, float rot_thrust)
{
    main_thrust_amount = main_thrust;
    rot_thrust_amount = rot_thrust;
}
void Rocket::applyPhysics()
{
    float scale = 0.01;
    float lin_friction = 0.99f;
    float rot_friction = 0.55f;
    
    rotation_dt_dt = scale * rot_thrust_amount * ROT_THRUST_MAG - 0.2 * rotation;
    rotation_dt = rot_friction * rotation_dt + rotation_dt_dt;
    rotation = rotation + rotation_dt;
    while (rotation > M_PI)
        rotation -= 2 * M_PI;
    while (rotation < -M_PI)
        rotation += 2 * M_PI;
    array rotation_matrix ({cos(rotation), sin(rotation),
                            -sin(rotation), cos(rotation)}, {2, 2});
    
    acceleration = scale * (gravity + main_thrust_amount * matmul(main_thrust, rotation_matrix));
    velocity = lin_friction * velocity + acceleration;
    position = position + velocity;
    position.eval();
    
    // Ground
    float* pos_data = position.data<float>();
    pos_data[0] = std::clamp(pos_data[0], -500.0f, 500.0f);
    if (pos_data[1] > 600.0f)
        pos_data[1] = 600.0f;
    if (pos_data[1] < 0.0f)
    {
        float ground_force = 0.0f - pos_data[1];
        
        // Collision
        pos_data[1] = 0.0f;
        velocity.eval();
        float* vel_data = velocity.data<float>();
        vel_data[1] = 0.0f;
        
        // Friction
        float ground_friction = 1.0f / (1.0f + ground_force);
        vel_data[0] = vel_data[0] * ground_friction;
    }
}

void Rocket::draw()
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (viewport == nullptr)
        return;
    
    float* data = position.data<float>();
    float x = data[0] + viewport->Size.x * 0.5f;
    float y = viewport->Size.y - (data[1] + 50);
    float theta = rotation;
    float r = 12.0f;
    
    float pi = M_PI;
    float PIo3 = 2 * pi * 0.4f;
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    draw->AddRectFilled({0, viewport->Size.y - 50}, {viewport->Size.x, viewport->Size.y}, IM_COL32(45, 45, 45, 255));
    draw->AddTriangleFilled({x - (0.8f + main_thrust_amount) * r * sin(theta + pi), y - (0.8f + main_thrust_amount) * r * cos(theta + pi)},
                            {x - r * sin(theta - PIo3), y - r * cos(theta - PIo3)},
                            {x - r * sin(theta + PIo3), y - r * cos(theta + PIo3)}, IM_COL32(245, 80, 15, 255));
    
    draw->AddTriangleFilled({x - r * sin(theta), y - r * cos(theta)},
                            {x - r * sin(theta + PIo3), y - r * cos(theta + PIo3)},
                            {x - r * sin(theta - PIo3), y - r * cos(theta - PIo3)}, IM_COL32(255, 255, 255, 255));
}

Rocket_HardCodedOptimalFunctionApproximation::Rocket_HardCodedOptimalFunctionApproximation()
{
    model.addLayer(Layer(2, 8));
    model.addLayer(Layer(8, 4));
    model.addLayer(Layer(4, 2));
}

void Rocket_HardCodedOptimalFunctionApproximation::computeControls(ImVec2 mouse_pos)
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (viewport == nullptr)
        return;
    
    // Get Environment
    mouse_pos.y = viewport->Size.y - mouse_pos.y - 50;
    mouse_pos.x = mouse_pos.x - viewport->Size.x * 0.5;
    array target_pos = array({mouse_pos.x, mouse_pos.y}, {1, 2});
    array relative_pos = target_pos - position;
    relative_pos.eval();
    
    // Forward
    array input = relative_pos;
    std::cout << "Input: " << input << "\n";
    array output = model.forward(input);
    output.eval();
    std::cout << "Output: " << output << "\n";
    
    float* output_data = output.data<float>();
    main_thrust_amount = output_data[0];
    rot_thrust_amount = 2.0 * (output_data[1] - 0.5f);
    
    // Heuristic Gradient
    array better_output = (relative_pos / max(abs(relative_pos))) * array({-0.5, 0.5}, output.shape()) + array({0.5, 0.5}, output.shape());
    better_output.eval();
    float* better_data = better_output.data<float>();
    better_output = array({better_data[1], better_data[0]}, {1, 2});
    std::cout << "Better Output: " << better_output << "\n";
    auto loss_fn = [&](array output)
    {
        return sum(square(better_output - output));
    };
    auto grad_fn = grad(loss_fn);
    auto grad = grad_fn(output);
    
    // Backward
    model.backward(grad, 1.1f);
}

Rocket_StatePredictorActionGenerator::Rocket_StatePredictorActionGenerator() :
Agent(4, 2, [](const array& state)
            {
                return 1.0f / (1.0f + sum(square(split(state, 2, 1)[0])));
            })
{
    /*
    // Pre-Compute Possible Actions
    for (float thrust = 0.0f; thrust <= 1.0f; thrust += 0.05f)
        for (float turn = 0.0f; turn <= 1.0f; turn += 0.05f)
            actions.push_back(array({thrust, turn}, {1, 2}));
    for (array& action : actions)
        action.eval();*/
}

void Rocket_StatePredictorActionGenerator::humanControlTraining(ImVec2 mouse_pos)
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (viewport == nullptr)
        return;
    
    // Get State
    mouse_pos.y = viewport->Size.y - mouse_pos.y - 50;
    mouse_pos.x = mouse_pos.x - viewport->Size.x * 0.5;
    array target_pos = array({mouse_pos.x, mouse_pos.y}, {1, 2});
    array relative_pos = target_pos - position;
    relative_pos.eval();
    array current_state = concatenate({relative_pos, velocity}, 1);
    
    array current_action({main_thrust_amount / 1.5f, rot_thrust_amount * 0.5f + 0.5f}, {1,2});
    trainOffline(current_state, prev_state, current_action, prev_action);
    commit_memory(current_state, current_action);
}

void Rocket_StatePredictorActionGenerator::computeControls(ImVec2 mouse_pos)
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (viewport == nullptr)
        return;
    
    // Get State
    mouse_pos.y = viewport->Size.y - mouse_pos.y - 50;
    mouse_pos.x = mouse_pos.x - viewport->Size.x * 0.5;
    array target_pos = array({mouse_pos.x, mouse_pos.y}, {1, 2});
    array relative_pos = target_pos - position;
    relative_pos.eval();
    array current_state = concatenate({relative_pos, velocity}, 1);
    
    // AI
    array current_action = process(current_state, true);
    
    // Act
    float* action_data = current_action.data<float>();
    main_thrust_amount = 1.5f * action_data[0];
    rot_thrust_amount = 2.0f * action_data[1] - 1.0f;
}
