#version 450
/* #include "gltf_common_inputs.frag" */
#include "gltf_common.frag"

layout(location = 0) out vec4 out_color;

void main() {
    vec4 base_color = get_final_color();
    base_color = vec4(base_color.rgb, 1.0);

    out_color = 0.0.xxxx
        + base_color
        + 0.0.xxxx;
}
