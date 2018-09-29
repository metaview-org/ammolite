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

layout(location = 0) in vec4 f_homogeneous_position;
layout(location = 1) in vec2 f_tex_coord;

layout(location = 0) out vec4 out_color;

vec4 get_base_color() {
    if (base_color_texture_provided) {
        vec4 texture_value = texture(
            sampler2D(base_color_texture, base_color_sampler),
            vec2(f_tex_coord)
        );

        return base_color_factor * texture_value;
    } else {
        return base_color_factor;
    }
}

void main() {
    vec4 base_color = get_base_color();
    base_color = vec4(base_color.rgb, 1.0);

    out_color = 0.0.xxxx
        + base_color
        + 0.0.xxxx;
}