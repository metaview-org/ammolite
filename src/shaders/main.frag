#version 450

layout(set = 0, binding = 0) uniform MainUBO {
    vec2 dimensions;
    mat4 model;
    mat4 view;
    mat4 projection;
};
layout(set = 0, binding = 1) uniform sampler2D screen_sampler;
layout(set = 0, binding = 2) uniform sampler2D tex;
layout(location = 0) in vec2 f_tex_coord;
layout(location = 0) out vec4 out_color;

void main() {
    vec2 uv = gl_FragCoord.xy / dimensions;
    out_color = vec4(0.0, 0.0, 0.0, 1.0)
        /* + texture(screen_sampler, uv) */
        + texture(tex, f_tex_coord);
}
