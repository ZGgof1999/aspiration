#include "grid.h"

Grid::Grid() { }
Grid::~Grid() { }

ImVec2 Grid::getTopLeft(float x, float y, ImVec2 center) { return {x*gridSize - halfGrid + padding + center.x, y*gridSize - halfGrid + padding + center.y}; }
ImVec2 Grid::getBottomRight(float x, float y, ImVec2 center) { return {x*gridSize + halfGrid - padding + center.x, y*gridSize + halfGrid - padding + center.y}; }

void Grid::render()
{
    // Controls
    ImGuiIO &io = ImGui::GetIO();
    ImGui::SetNextWindowSizeConstraints({200, 100}, {999999, 999999});
    ImGui::SetNextWindowPos({gridSize, gridSize});
    ImGui::Begin("Controls");
    
    ImGui::InputInt("Batches", &numBatches);
    ImGui::InputFloat("Learning Rate", &learningRate);
    if (ImGui::Button("Train"))
        dot.train(numBatches, learningRate);
    
    ImVec2 size = ImGui::GetMainViewport()->Size;
    ImVec2 center(size.x * 0.5, size.y * 0.5);
    if (io.MouseDown[0])
    {
        targetX = std::floor((io.MousePos.x - center.x + halfGrid) / gridSize);
        targetY = std::floor((io.MousePos.y - center.y + halfGrid) / gridSize);
    }
    targetX = std::clamp(targetX, -8.0f, 8.0f);
    targetY = std::clamp(targetY, -8.0f, 8.0f);
    
    ImGui::End();
    
    dot.act(targetX, targetY);
    
    // Paint
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    ImU32 color = IM_COL32(25, 25, 25, 255);
    
    for (float x = -8; x <= 8; x++)
        for (float y = -8; y <= 8; y++)
            draw->AddRectFilled(getTopLeft(x, y, center),
                                getBottomRight(x, y, center),
                                color);
    draw->AddRectFilled(getTopLeft(targetX, targetY, center), getBottomRight(targetX, targetY, center), IM_COL32(65, 65, 65, 255));
    draw->AddCircleFilled({dot.x * gridSize + center.x, dot.y * gridSize + center.y}, halfGrid - 2 * padding, IM_COL32(205, 25, 25, 255));
    draw->AddCircle(io.MousePos, 10, IM_COL32_WHITE);
}
