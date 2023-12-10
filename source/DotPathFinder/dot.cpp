#include "dot.h"

Dot::Dot() :
    l1(2, 8),
    l2(8, 5),
    l3(5, 5)
{
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
            array output = l1.forward(input);
            output = l2.forward(output);
            output = l3.forward(output);
            output.eval();
            
            auto loss_fn = [&](array output)
            {
                return sum(square((correct / max(correct)) - (output / max(output))));
            };
            loss += loss_fn(output).item<float>();
            auto grad_fn = grad(loss_fn);
            auto grad = grad_fn(output);
            
            grad = l3.backward(grad, learning_rate);
            grad = l2.backward(grad, learning_rate);
            grad = l1.backward(grad, learning_rate);
        }
        std::cout << "Batch: " << batch + 1 << " / " << num_batches << ", Loss: " << loss / trainingData.size() << "\n";
    }
}
void Dot::act(float target_x, float target_y)
{
    // AI Model
    array input({(target_x - x) / 8.0f, (target_y - y) / 8.0f}, {1, 2});
    array output = l1.forward(input);
    output = l2.forward(output);
    output = l3.forward(output);
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
