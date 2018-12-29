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

layout(set = 2, binding = 0) uniform MaterialUBO {
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
    bool base_color_texture_provided;
    float alpha_cutoff;
};
layout(set = 2, binding = 1) uniform texture2D base_color_texture;
layout(set = 2, binding = 2) uniform sampler base_color_sampler;

layout(set = 3, binding = 0, input_attachment_index = 0) uniform subpassInput attachment_accumulation;
layout(set = 3, binding = 1, input_attachment_index = 1) uniform subpassInput attachment_revealage;

layout(location = 0) in vec4 f_homogeneous_position;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec4 f_tangent;
layout(location = 3) in vec2 f_tex_coord;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 accumulation = subpassLoad(attachment_accumulation);
    float revealage = subpassLoad(attachment_revealage).r;
    out_color = vec4(accumulation.rgb / accumulation.a, revealage);
}
