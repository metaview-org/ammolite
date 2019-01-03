#version 450

#define PROJECT(vector4) (vector4.w == 0 ? vector4.xyz : (vector4.xyz / vector4.w))
#define GRAM_SCHMIDT(a, b) (a - (b) * dot((a), (b)))

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

layout(location = 0) out vec3 f_position;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec4 f_tangent;
layout(location = 3) out vec2 f_tex_coord;

void main() {
    // Ensure the normal and tangent are orthonormal
    vec3 normalized_normal = normalize(normal);
    vec3 normalized_tangent = normalize(tangent.xyz);
    vec3 corrected_tangent = GRAM_SCHMIDT(normalized_tangent, normalized_normal);

    // Apply the transformation of primitives to view space
    vec4 view_position = view * model * matrix * vec4(position, 1.0);
    vec4 view_normal = transpose(inverse(view * model * matrix)) * vec4(normalized_normal, 0.0);
    vec4 view_tangent = view * model * matrix * vec4(corrected_tangent, 0.0);

    // Ensure the normal and tangent are orthonormal, again
    vec3 normalized_projected_view_normal = normalize(PROJECT(view_normal));
    vec3 projected_view_tangent = PROJECT(view_tangent);
    vec3 corrected_view_tangent = GRAM_SCHMIDT(projected_view_tangent, normalized_projected_view_normal);

    f_position = PROJECT(view_position);
    f_normal = normalized_projected_view_normal;
    f_tangent = vec4(normalize(corrected_view_tangent), tangent.w);
    f_tex_coord = tex_coord;
    gl_Position = projection * view_position;
}
