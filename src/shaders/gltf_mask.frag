#version 450
#include "gltf_common.frag"

layout(set = 0, binding = 0) uniform SceneUBO {
    vec2 dimensions;
    vec3 camera_position;
    mat4 model;
    mat4 view;
    mat4 projection;
};

layout(set = 1, binding = 0) uniform NodeUBO {
    mat4 matrix;
};

layout(set = 2, binding = 0) uniform MaterialUBO {
    float alpha_cutoff;

    bool base_color_texture_provided;
    vec4 base_color_factor;

    bool metallic_roughness_texture_provided;
    vec2 metallic_roughness_factor;

    bool normal_texture_provided;
    float normal_texture_scale;

    bool occlusion_texture_provided;
    float occlusion_strength;

    bool emissive_texture_provided;
    vec3 emissive_factor;
};
layout(set = 2, binding =  1) uniform texture2D base_color_texture;
layout(set = 2, binding =  2) uniform sampler base_color_sampler;
layout(set = 2, binding =  3) uniform texture2D metallic_roughness_texture;
layout(set = 2, binding =  4) uniform sampler metallic_roughness_sampler;
layout(set = 2, binding =  5) uniform texture2D normal_texture;
layout(set = 2, binding =  6) uniform sampler normal_sampler;
layout(set = 2, binding =  7) uniform texture2D occlusion_texture;
layout(set = 2, binding =  8) uniform sampler occlusion_sampler;
layout(set = 2, binding =  9) uniform texture2D emissive_texture;
layout(set = 2, binding = 10) uniform sampler emissive_sampler;

layout(location = 0) in vec3 f_world_position;
layout(location = 1) in vec3 f_world_normal;
layout(location = 2) in vec4 f_world_tangent;
layout(location = 3) in vec2 f_tex_coord;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 base_color = GET_FINAL_COLOR();

    if (base_color.a < alpha_cutoff) {
        discard;
    } else {
        base_color = vec4(base_color.rgb, 1.0);
    }

    out_color = 0.0.xxxx
        + base_color
        + 0.0.xxxx;
}
