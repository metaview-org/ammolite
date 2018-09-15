#version 450

layout(set = 0, binding = 0) uniform SceneUBO {
    vec2 dimensions;
    mat4 model;
    mat4 view;
    mat4 projection;
};
layout(set = 0, binding = 1) uniform sampler2D screen_sampler;

layout(set = 1, binding = 0) uniform NodeUBO {
    mat4 matrix;
};

layout(set = 2, binding = 0) uniform MaterialUBO {
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
};
layout(set = 2, binding = 1) uniform sampler2D base_color_texture;

layout(location = 0) in vec4 f_homogeneous_position;
layout(location = 1) in vec2 f_tex_coord;

layout(location = 0) out vec4 out_color;

void main() {
    vec2 uv = gl_FragCoord.xy / dimensions;
    out_color = 0.0.xxxx
        /* + texture(base_color_texture, uv) */
        /* + vec4(f_homogeneous_position.xyz / f_homogeneous_position.w, 1.0); */
        /* + texture(screen_sampler, uv) */
        + texture(base_color_texture, vec2(f_tex_coord))
        + 0.0.xxxx;
}
