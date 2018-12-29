vec4 get_base_color(in bool base_color_texture_provided,
                    in vec4 base_color_factor,
                    in texture2D base_color_texture,
                    in sampler base_color_sampler,
                    in vec2 f_tex_coord) {
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

vec4 get_final_color(in vec3 frag_position,
                     in vec3 normal,
                     in bool base_color_texture_provided,
                     in vec4 base_color_factor,
                     in texture2D base_color_texture,
                     in sampler base_color_sampler,
                     in vec2 f_tex_coord) {
    vec4 base_color = get_base_color(base_color_texture_provided,
                                     base_color_factor,
                                     base_color_texture,
                                     base_color_sampler,
                                     f_tex_coord);
    vec3 light_dir = vec3(0.0, 0.0, -1.0);
    float weight = clamp(dot(normal, light_dir), 0.0, 1.0);

    return base_color * weight;
}
