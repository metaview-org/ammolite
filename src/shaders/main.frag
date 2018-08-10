#version 450

layout(set = 0, binding = 0) uniform MainDimensions {
    vec2 main_dimensions;
};
layout(set = 0, binding = 1) uniform sampler2D screen_sampler;
layout(location = 0) out vec4 out_color;

void main() {
    vec2 uv = gl_FragCoord.xy / main_dimensions;
    out_color = vec4(1.0, 0.0, 0.0, 1.0)
        + texture(screen_sampler, uv)
        + vec4(0.0, uv, 0.0);
}
