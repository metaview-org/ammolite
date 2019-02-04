#version 450
/* #include "gltf_common_inputs.frag" */
#include "gltf_common.frag"

layout(set = 3, binding = 0, input_attachment_index = 0) uniform subpassInput attachment_accumulation;
layout(set = 3, binding = 1, input_attachment_index = 1) uniform subpassInput attachment_revealage;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 accumulation = subpassLoad(attachment_accumulation);
    float revealage = subpassLoad(attachment_revealage).r;
    out_color = vec4(accumulation.rgb / accumulation.a, revealage);
}
