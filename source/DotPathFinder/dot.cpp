#include "dot.h"

Dot::Dot()
{
    model.addLayer(Layer(2, 8));
    model.addLayer(Layer(8, 5));
    model.addLayer(Layer(5, 5));
    
    for(float x = -16; x <= 16; x++)
        for (float y = -16; y <= 16; y++)
            trainingData.push_back({
                {x, y},
                {   0.1f,
                    std::max(0.0f, (-x / 8.0f)),
                    std::max(0.0f, (x / 8.0f)),
                    std::max(0.0f, (y / 8.0f)),
                    std::max(0.0f, (-y / 8.0f))
                }
            });
}
Dot::~Dot() { }

void Dot::train(int num_batches, float learning_rate)
{
    for (int batch = 0; batch < num_batches; batch++)
    {
        float loss = 0.0f;
        for (TrainingData& td : trainingData)
        {
            array correct(td.y.begin(), {1,5});
            array input(td.x.begin(), {1,2});
            array output = model.forward(input);
            output.eval();
            
            auto loss_fn = [&](array output)
            {
                return sum(square((correct / max(correct)) - (output / max(output))));
            };
            loss += loss_fn(output).item<float>();
            auto grad_fn = grad(loss_fn);
            auto grad = grad_fn(output);
            
            model.backward(grad, learning_rate);
        }
        std::cout << "Batch: " << batch + 1 << " / " << num_batches << ", Loss: " << loss / trainingData.size() << "\n";
    }
}
void Dot::act(float target_x, float target_y)
{
    // AI Model
    array input({(target_x - x) / 8.0f, (target_y - y) / 8.0f}, {1, 2});
    array output = model.forward(input);
    output.eval();
    
    float* data = output.data<float>();
    float max = 0;
    int action = 0;
    for (int a = 0; a < 5; a++)
        if (data[a] > max)
        {
            max = data[a];
            action = a;
        }
    if (action == Action::left)
        x = std::clamp(x-1, -8.0f, 8.0f);
    else if (action == Action::right)
        x = std::clamp(x+1, -8.0f, 8.0f);
    else if (action == Action::up)
        y = std::clamp(y+1, -8.0f, 8.0f);
    else if (action == Action::down)
        y = std::clamp(y-1, -8.0f, 8.0f);
}

OnlineDot::OnlineDot() :
    prev_state(zeros({1,2})),
    prev_action({0.1f, 0.0f, 0.0f, 0.0f, 0.0f}, {1, 5})
{
    int state_size = 2;
    int action_size = 5;
    state_predictor.addLayer(Layer(state_size + action_size, 16));
    state_predictor.addLayer(Layer(16, 8));
    state_predictor.addLayer(Layer(8, state_size, false));
    
    action_generator.addLayer(Layer(state_size * 2, 16));
    action_generator.addLayer(Layer(16, 8));
    action_generator.addLayer(Layer(8, action_size, false));
    
    actions.push_back(array({1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {1, 5}));
    actions.push_back(array({0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, {1, 5}));
    actions.push_back(array({0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, {1, 5}));
    actions.push_back(array({0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, {1, 5}));
    actions.push_back(array({0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {1, 5}));
    for (array& action : actions)
        action.eval();
}
OnlineDot::~OnlineDot() { }

float OnlineDot::getStateScore(array state)
{
    state.eval();
    float* state_data = state.data<float>();
    float dist = state_data[0] * state_data[0] + state_data[1] * state_data[1];
    return 1.0f / (1.0f + dist);
}

void OnlineDot::act(float target_x, float target_y, float state_rate, float action_rate)
{
    std::cout << "\n" << "\n" << "\n";
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (viewport == nullptr)
        return;
    
    // Get State
    array target_pos = array({target_x, target_y}, {1, 2});
    array position = array({x, y}, {1, 2});
    array current_state = (target_pos - position) / 8.0f;
    current_state.eval();
    std::cout << "Current State: " << current_state << "\n";
    
    // Train State Prediction
    array prev_predicted_state = state_predictor.forward(concatenate({prev_state, prev_action}, 1));
    auto state_predictor_loss_fn = [&](array output)
    {
        return sum(square(current_state - output));
    };
    auto state_predictor_grad_fn = grad(state_predictor_loss_fn);
    auto state_predictor_grad = state_predictor_grad_fn(prev_predicted_state);
    state_predictor.backward(state_predictor_grad, state_rate);
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
    action_generator.backward(action_generator_grad, action_rate);
    float ag_loss = action_generator_loss_fn(prev_generated_action).item<float>();
    std::cout << "AG Loss: " << ag_loss << "\n";
    
    // Choose Best Action
    int selected_action = Action::stay;
    float score = 0.0f;
    for (int action = 0; action < actions.size(); action++)
    {
        if ((action == Action::left && x == -8) || (action == Action::stay) ||
            (action == Action::right && x == 8) ||
            (action == Action::up && y == 8) ||
            (action == Action::down && y == -8))
            continue;
        array predicted_state = state_predictor.forward(concatenate({current_state, actions[action]}, 1));
        float new_score = getStateScore(predicted_state);
        //std::cout << "a" << action + 1 << ": " << new_score << "\n";
        if (new_score > score)
        {
            selected_action = action;
            score = new_score;
        }
    }
    std::cout << "Best Action: " << selected_action + 1 << " / " << actions.size() << "\n";

    // Act
    if (selected_action == Action::left)
        x = std::clamp(x-1, -8.0f, 8.0f);
    else if (selected_action == Action::right)
        x = std::clamp(x+1, -8.0f, 8.0f);
    else if (selected_action == Action::up)
        y = std::clamp(y+1, -8.0f, 8.0f);
    else if (selected_action == Action::down)
        y = std::clamp(y-1, -8.0f, 8.0f);
    
    // Memory
    prev_state = current_state;
    prev_action = actions[selected_action];
}
