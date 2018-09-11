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
/* layout(location = 1) in vec2 tex_coord; */
/* layout(location = 0) out vec2 f_tex_coord; */
layout(location = 0) out vec4 f_homogeneous_position;

void main() {
    /* gl_Position = vec4(position, 1.0); */
    gl_Position = projection * view * model * matrix * vec4(position, 1.0);
    /* gl_Position = projection * view * model * vec4(position, 1.0); */
    /* f_tex_coord = tex_coord; */
    f_homogeneous_position = gl_Position;
}
