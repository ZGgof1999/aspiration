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
    prev_state(zeros({1,4})),
    prev_action({0.0f, 0.5f}, {1, 2})
{
    int state_size = 4;
    int action_size = 2;
    state_predictor.addLayer(Layer(state_size + action_size, 16));
    state_predictor.addLayer(Layer(16, 8));
    state_predictor.addLayer(Layer(8, state_size, false));
    
    action_generator.addLayer(Layer(state_size * 2, 16));
    action_generator.addLayer(Layer(16, 8));
    action_generator.addLayer(Layer(8, action_size));
    
    // Pre-Compute Possible Actions
    for (float thrust = 0.0f; thrust <= 1.0f; thrust += 0.05f)
        for (float turn = 0.0f; turn <= 1.0f; turn += 0.05f)
            actions.push_back(array({thrust, turn}, {1, 2}));
    for (array& action : actions)
        action.eval();
}

float Rocket_StatePredictorActionGenerator::getStateScore(array state)
{
    state.eval();
    float* state_data = state.data<float>();
    float dist = state_data[0] * state_data[0] + state_data[1] * state_data[1];
    return 1.0f / (1.0f + dist);
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
    current_state.eval();
    std::cout << "Current State: " << current_state << "\n";
    
    // Train State Prediction
    array prev_predicted_state = state_predictor.forward(concatenate({prev_state, 1000 * prev_action}, 1));
    auto state_predictor_loss_fn = [&](array output)
    {
        return sum(square(current_state - output));
    };
    auto state_predictor_grad_fn = grad(state_predictor_loss_fn);
    auto state_predictor_grad = state_predictor_grad_fn(prev_predicted_state);
    state_predictor.backward(state_predictor_grad, 0.001f);
    float sp_loss = state_predictor_loss_fn(prev_predicted_state).item<float>();
    std::cout << "SP Loss: " << sp_loss << "\n";
    
    // Train Action Generator
    array prev_generated_action = action_generator.forward(concatenate({prev_state, current_state}, 1));
    auto action_generator_loss_fn = [&](array output)
    {
        return sum(square(prev_action - output));
    };
    auto action_generator_grad_fn = grad(action_generator_loss_fn);
    auto action_generator_grad = action_generator_grad_fn(prev_generated_action);
    action_generator.backward(action_generator_grad, 0.01f);
    float ag_loss = action_generator_loss_fn(prev_generated_action).item<float>();
    std::cout << "AG Loss: " << ag_loss << "\n";
    
    // Choose Best Action
    int s = sqrt(actions.size());
    int selected_action = s * (s-3.5f);
    float score = 0.0f;
    int random_action = static_cast<int>(sp_loss * actions.size()) % actions.size();
    if (sp_loss > 100.0f)
        selected_action = random_action;
    else
        for (int action = 0; action < actions.size(); action++)
        {
            array predicted_state = state_predictor.forward(concatenate({current_state,  1000 * actions[action]}, 1));
            float new_score = getStateScore(predicted_state);
            std::cout << "a" << action + 1 << ": " << new_score << "\n";
            if (new_score > score)
            {
                selected_action = action;
                score = new_score;
            }
        }
    std::cout << "Best Action: " << selected_action + 1 << " / " << actions.size() << "\n";

    // Act
    float* action_data = actions[selected_action].data<float>();
    main_thrust_amount = 1.5f * action_data[0];
    rot_thrust_amount = 2.0f * action_data[1] - 1.0f;
    std::cout << "Action: " << main_thrust_amount << ", " << rot_thrust_amount << "\n";
    
    // Memory
    prev_state = current_state;
    prev_action = actions[selected_action];
}
