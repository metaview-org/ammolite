#version 450
/* #include "gltf_common_inputs.frag" */
#include "gltf_common.frag"

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
    vec4 base_color = get_final_color();
    // Without premultiplication:
    vec4 premultiplied_alpha_color = vec4(base_color.rgb, 1.0);
    // With premultiplication:
    /* vec4 premultiplied_alpha_color = vec4(base_color.rgb * base_color.a, base_color.a); */

    // Sums up both the numerator and the denominator of the WBOIT expression
    out_accumulation_src = premultiplied_alpha_color * weight_function(gl_FragCoord, base_color.a);
    out_revealage_src = base_color.aaaa;
}
