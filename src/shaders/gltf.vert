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
    // Applying both transformations at once using the following line of code
    // wouldn't work for some reason.
    /* vec4 world_normal = transpose(inverse(model * matrix)) * vec4(normalized_normal, 0.0); */
    /* vec4 world_normal = inverse(model) * inverse(matrix) * vec4(normalized_normal, 0.0); */
    vec4 world_normal = inverse(matrix * model * y_inversion) * vec4(normalized_normal, 0.0);
    vec4 world_tangent = y_inversion * model * matrix * vec4(corrected_tangent, 0.0);

    // Ensure the normal and tangent are orthonormal, again
    vec3 normalized_projected_world_normal = normalize(PROJECT(world_normal));
    vec3 projected_world_tangent = PROJECT(world_tangent);
    vec3 corrected_world_tangent = GRAM_SCHMIDT(projected_world_tangent, normalized_projected_world_normal);

    f_world_position = PROJECT(world_position);
    f_world_normal = normalized_projected_world_normal;
    f_world_tangent = vec4(normalize(corrected_world_tangent), tangent.w);
    f_tex_coord = tex_coord;
    gl_Position = y_inversion * projection * view * world_position;
}
