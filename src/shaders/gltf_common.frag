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
