#include "Agent.h"

Agent::Agent(int state_size, int action_size, std::function<array(const array&)> score_state) :
    prev_state(zeros({1,state_size})),
    prev_action(zeros({1,action_size})),
    score_state_fn(score_state)
{
    state_predictor.addLayer(Layer(state_size + action_size, 12));
    state_predictor.addLayer(Layer(12, 8));
    state_predictor.addLayer(Layer(8, 8));
    state_predictor.addLayer(Layer(8, state_size, false));
    
    action_generator.addLayer(Layer(state_size * 2, 12));
    action_generator.addLayer(Layer(12, 8));
    action_generator.addLayer(Layer(8, 4));
    action_generator.addLayer(Layer(4, action_size));
}
Agent::~Agent() { }

float Agent::train_state_predictor(array current_state, float learning_rate)
{
    array prev_predicted_state = state_predictor.forward(concatenate({prev_state, prev_action}, 1));
    auto state_predictor_loss_fn = [&](array output)
    {
        return sum(square(current_state - output));
    };
    auto state_predictor_grad_fn = grad(state_predictor_loss_fn);
    auto state_predictor_grad = state_predictor_grad_fn(prev_predicted_state);
    state_predictor.backward(state_predictor_grad, learning_rate);
    float sp_loss = state_predictor_loss_fn(prev_predicted_state).item<float>();
    return sp_loss;
}
float Agent::train_action_generator(array current_state, float learning_rate)
{
    array prev_generated_action = action_generator.forward(concatenate({prev_state, current_state}, 1));
    auto action_generator_loss_fn = [&](array output)
    {
        return sum(square(prev_action - output));
    };
    auto action_generator_grad_fn = grad(action_generator_loss_fn);
    auto action_generator_grad = action_generator_grad_fn(prev_generated_action);
    action_generator.backward(action_generator_grad, learning_rate);
    float ag_loss = action_generator_loss_fn(prev_generated_action).item<float>();
    return ag_loss;
}
void Agent::trainOffline(array sn1, array sn0, array an1, array an0)
{
    commit_memory(sn0, an0);
    train_state_predictor(sn1, 0.02f);
    train_action_generator(sn1, 0.02f);
}

array Agent::generate_action(array current_state, float aspiration_rate)
{
    auto state_score_grad_fn = grad(score_state_fn);
    auto state_score_grad = state_score_grad_fn(current_state);
    std::cout << "Grad: " << state_score_grad << "\n";
    array chosen_action = prev_action;
    float score = 0.0f;
    for (float aspiration_rate = 0.01f; aspiration_rate <= 100.0f; aspiration_rate *= -2.0f)
    {
        auto ideal_state = current_state + aspiration_rate * state_score_grad;
        array generated_action = action_generator.forward(concatenate({current_state, ideal_state}, 1));
        array realistic_state = state_predictor.forward(concatenate({current_state, generated_action}, 1));
        float new_score = score_state_fn(realistic_state).item<float>();
        if (new_score > score)
        {
            chosen_action = generated_action;
            score = new_score;
        }
    }
    std::cout << "Score: " << score << "\n";
    return chosen_action;
}

array Agent::select_action(array current_state)
{
    // Choose Best Action
    int selected_action = 0;
    float score = 0.0f;
    for (int action = 0; action < actions.size(); action++)
    {
        array predicted_state = state_predictor.forward(concatenate({current_state, actions[action]}, 1));
        float new_score = score_state_fn(predicted_state).item<float>();
        if (new_score > score)
        {
            selected_action = action;
            score = new_score;
        }
    }
    return actions[selected_action];
}

void Agent::commit_memory(array current_state, array current_action)
{
    prev_state = current_state;
    prev_action = current_action;
}

array Agent::process(array current_state, bool debug)
{
    if (debug)
        std::cout << "\n" << "\n" << "\n";
    current_state.eval();
    float sp_loss = train_state_predictor(current_state, 0.002f);
    float ag_loss = train_action_generator(current_state, 0.002f);
    array current_action = generate_action(current_state, 0.1f);
    //array current_action = select_action(current_state);
    //array current_action = array({0.8, 0.5}, {1, 2});
    commit_memory(current_state, current_action);
    
    if (debug)
    {
        std::cout
        << "State: " << current_state << "\n"
        << "Action: " << current_action << "\n"
        << "SP Loss: " << sp_loss << "\n"
        << "AG Loss: " << ag_loss << "\n";
    }
    return current_action;
}
