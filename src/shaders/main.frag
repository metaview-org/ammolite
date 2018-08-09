#version 450

layout(set = 0, binding = 0, location = 0) uniform texture2D screen_texture;
layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
