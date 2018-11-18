#version 450
#include "gltf_common.frag"

layout(set = 0, binding = 0) uniform SceneUBO {
    vec2 dimensions;
    mat4 model;
    mat4 view;
    mat4 projection;
};

layout(set = 1, binding = 0) uniform NodeUBO {
    mat4 matrix;
};

layout(set = 2, binding = 0) uniform MaterialUBO {
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
    bool base_color_texture_provided;
    float alpha_cutoff;
};
layout(set = 2, binding = 1) uniform texture2D base_color_texture;
layout(set = 2, binding = 2) uniform sampler base_color_sampler;

layout(location = 0) in vec4 f_homogeneous_position;
layout(location = 1) in vec2 f_tex_coord;

layout(location = 0) out vec4 out_accumulation_src;
layout(location = 1) out vec4 out_revealage_src; //FIXME

float linearize_z(vec4 homogeneous_position) {
    mat4 inverse_projection = inverse(projection);

    // Won't work, because the position is normalized
    return (inverse_projection * homogeneous_position).z;
}

float weight_function_inverse(vec4 homogeneous_position, float alpha) {
    float z = linearize_z(homogeneous_position);
    return 1.0 / (z + 1.0);
}

float weight_function_paper_eq_7(vec4 homogeneous_position, float alpha) {
    float z = linearize_z(homogeneous_position);
    float base_mem1 = abs(z) / 5.0;
    float mem1 = base_mem1 * base_mem1;
    float base_mem2 = abs(z) / 200.0;
    float mem2 = base_mem2 * base_mem2 * base_mem2 * base_mem2 * base_mem2 * base_mem2;
    return alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + mem1 + mem2)));
}

float weight_function_paper_eq_8(vec4 homogeneous_position, float alpha) {
    float z = linearize_z(homogeneous_position);
    float base_mem1 = abs(z) / 10.0;
    float mem1 = base_mem1 * base_mem1 * base_mem1;
    float base_mem2 = abs(z) / 200.0;
    float mem2 = base_mem2 * base_mem2 * base_mem2 * base_mem2 * base_mem2 * base_mem2;
    return alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + mem1 + mem2)));
}

float weight_function_paper_eq_9(vec4 homogeneous_position, float alpha) {
    float z = linearize_z(homogeneous_position);
    float base_mem = abs(z) / 200.0;
    float mem = base_mem * base_mem * base_mem * base_mem;
    return alpha * max(1e-2, min(3e3, 0.03 / (1e-5 + mem)));
}

float weight_function_paper_eq_10(vec4 homogeneous_position, float alpha) {
    float one_minus_z = 1.0 - homogeneous_position.z;
    float cubed = one_minus_z * one_minus_z * one_minus_z;
    return alpha * max(1e-2, 3e3 * cubed);
}

float weight_function(vec4 homogeneous_position, float alpha) {
    return weight_function_paper_eq_8(homogeneous_position, alpha);
}

void main() {
    vec4 base_color = get_base_color(base_color_texture_provided,
                                     base_color_factor,
                                     base_color_texture,
                                     base_color_sampler,
                                     f_tex_coord);
    // Without premultiplication:
    vec4 premultiplied_alpha_color = vec4(base_color.rgb, 1.0);
    // With premultiplication:
    /* vec4 premultiplied_alpha_color = vec4(base_color.rgb * base_color.a, base_color.a); */

    // Sums up both the numerator and the denominator of the WBOIT expression
    out_accumulation_src = premultiplied_alpha_color * weight_function(gl_FragCoord, base_color.a);
    out_revealage_src = base_color.aaaa;
}
