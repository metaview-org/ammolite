#version 450

layout(set = 0, binding = 0) uniform SceneUBO {
    vec2 dimensions;
    mat4 model;
    mat4 view;
    mat4 projection;
};

layout(set = 1, binding = 0) uniform NodeUBO {
    mat4 matrix;
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 tex_coord;

layout(location = 0) out vec4 f_homogeneous_position;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec4 f_tangent;
layout(location = 3) out vec2 f_tex_coord;

void main() {
    gl_Position = projection * view * model * matrix * vec4(position, 1.0);
    /* gl_Position = projection * view * model * vec4(position, 1.0); */

    f_homogeneous_position = gl_Position;
    f_normal = normal;
    f_tex_coord = tex_coord;
}
