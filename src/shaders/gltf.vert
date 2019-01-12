#version 450
#include "gltf_common.h"

layout(set = 0, binding = 0) uniform SceneUBO {
    float time_elapsed;
    vec2 dimensions;
    vec3 camera_position;
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

layout(location = 0) out vec3 f_world_position;
layout(location = 1) out vec3 f_world_normal;
layout(location = 2) out vec4 f_world_tangent;
layout(location = 3) out vec2 f_tex_coord;

const mat4 y_inversion = mat4(
    1.0,  0.0,  0.0,  0.0,
    0.0, -1.0,  0.0,  0.0,
    0.0,  0.0,  1.0,  0.0,
    0.0,  0.0,  0.0,  1.0
);

void main() {
    // Ensure the normal and tangent are orthonormal
    vec3 normalized_normal = normalize(normal);
    vec3 normalized_tangent = normalize(tangent.xyz);
    vec3 corrected_tangent = GRAM_SCHMIDT(normalized_tangent, normalized_normal);

    // Apply the transformation of primitives to view space
    vec4 world_position = model * matrix * vec4(position, 1.0);
    // Note: trying to invert and transpose the 4x4 matrix results in artifacts
    vec3 world_normal = normalize(transpose(inverse(mat3(model * matrix))) * normalized_normal);
    vec3 world_tangent = mat3(model * matrix) * corrected_tangent.xyz;

    // Ensure the normal and tangent are orthonormal, again
    vec3 corrected_world_tangent = normalize(GRAM_SCHMIDT(world_tangent, world_normal));

    f_world_position = PROJECT(world_position);
    f_world_normal = world_normal;
    f_world_tangent = vec4(corrected_world_tangent, tangent.w);
    f_tex_coord = tex_coord;
    gl_Position = y_inversion * projection * view * world_position;
}
