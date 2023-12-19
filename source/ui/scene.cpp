#include "scene.h"

Scene::Scene() { }
Scene::~Scene() { }

void Scene::render()
{
    //grid.render();
    environment.render();
}
