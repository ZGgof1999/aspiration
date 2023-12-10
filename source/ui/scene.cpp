#include "scene.h"

Scene::Scene() { }
Scene::~Scene() { }

void Scene::render()
{
    environment.render();
}
