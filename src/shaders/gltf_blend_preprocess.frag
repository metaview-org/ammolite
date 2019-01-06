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

layout(location = 0) out vec4 out_accumulation_src;
layout(location = 1) out vec4 out_revealage_src; //FIXME

float weight_function_inverse(vec4 position, float alpha) {
    float z = position.z;
    return 1.0 / (z + 1.0);
}

float weight_function_paper_eq_7(vec4 position, float alpha) {
    float z = position.z;
    float base_mem1 = abs(z) / 5.0;
    float mem1 = base_mem1 * base_mem1;
    float base_mem2 = abs(z) / 200.0;
    float mem2 = base_mem2 * base_mem2 * base_mem2 * base_mem2 * base_mem2 * base_mem2;
    return alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + mem1 + mem2)));
}

float weight_function_paper_eq_8(vec4 position, float alpha) {
    float z = position.z;
    float base_mem1 = abs(z) / 10.0;
    float mem1 = base_mem1 * base_mem1 * base_mem1;
    float base_mem2 = abs(z) / 200.0;
    float mem2 = base_mem2 * base_mem2 * base_mem2 * base_mem2 * base_mem2 * base_mem2;
    return alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + mem1 + mem2)));
}

float weight_function_paper_eq_9(vec4 position, float alpha) {
    float z = position.z;
    float base_mem = abs(z) / 200.0;
    float mem = base_mem * base_mem * base_mem * base_mem;
    return alpha * max(1e-2, min(3e3, 0.03 / (1e-5 + mem)));
}

float weight_function_paper_eq_10(vec4 position, float alpha) {
    float one_minus_z = 1.0 - gl_FragCoord.z;
    float cubed = one_minus_z * one_minus_z * one_minus_z;
    return alpha * max(1e-2, 3e3 * cubed);
}

float weight_function(vec4 position, float alpha) {
    return weight_function_paper_eq_8(position, alpha);
}

void main() {
    vec4 base_color = GET_FINAL_COLOR();
    // Without premultiplication:
    vec4 premultiplied_alpha_color = vec4(base_color.rgb, 1.0);
    // With premultiplication:
    /* vec4 premultiplied_alpha_color = vec4(base_color.rgb * base_color.a, base_color.a); */

    // Sums up both the numerator and the denominator of the WBOIT expression
    out_accumulation_src = premultiplied_alpha_color * weight_function(gl_FragCoord, base_color.a);
    out_revealage_src = base_color.aaaa;
}
