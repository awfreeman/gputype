#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;
#define PI 3.14159
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = (mat2(1, 0, 0,- 1)*position) + vec2(0.5, 0.5);
}