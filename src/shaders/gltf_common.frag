#include "gltf_common.h"

vec2 get_normalized_frag_coord(in vec2 dimensions) {
    return gl_FragCoord.xy / dimensions;
}

vec4 sample_base_color(in bool base_color_texture_provided,
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

vec2 sample_metallic_roughness(in bool metallic_roughness_texture_provided,
                               in vec2 metallic_roughness_factor,
                               in texture2D metallic_roughness_texture,
                               in sampler metallic_roughness_sampler,
                               in vec2 f_tex_coord) {
    if (metallic_roughness_texture_provided) {
        vec4 texture_value = texture(
            sampler2D(metallic_roughness_texture, metallic_roughness_sampler),
            vec2(f_tex_coord)
        );

        return metallic_roughness_factor * texture_value.gb;
    } else {
        return metallic_roughness_factor;
    }
}

float sample_occlusion(in bool occlusion_texture_provided,
                      in float occlusion_strength,
                      in texture2D occlusion_texture,
                      in sampler occlusion_sampler,
                      in vec2 f_tex_coord) {
    if (occlusion_texture_provided) {
        vec4 texture_value = texture(
            sampler2D(occlusion_texture, occlusion_sampler),
            vec2(f_tex_coord)
        );

        return occlusion_strength * texture_value.r;
    } else {
        return occlusion_strength;
    }
}

vec3 sample_emissive(in bool emissive_texture_provided,
                     in vec3 emissive_factor,
                     in texture2D emissive_texture,
                     in sampler emissive_sampler,
                     in vec2 f_tex_coord) {
    if (emissive_texture_provided) {
        vec4 texture_value = texture(
            sampler2D(emissive_texture, emissive_sampler),
            vec2(f_tex_coord)
        );

        return emissive_factor * texture_value.rgb;
    } else {
        return emissive_factor;
    }
}

vec3 sample_normal(in bool normal_texture_provided,
                   in float normal_texture_scale,
                   in texture2D normal_texture,
                   in sampler normal_sampler,
                   in vec2 f_tex_coord) {
    if (!normal_texture_provided) {
        return vec3(0.0, 0.0, 1.0);
    }

    vec3 normal_sample = texture(
        sampler2D(normal_texture, normal_sampler),
        vec2(f_tex_coord)
    ).xyz;

    vec3 scaled_normal = normalize((normal_sample * 2.0 - 1.0)
            * vec3(normal_texture_scale, normal_texture_scale, 1.0));

    return scaled_normal;
}

vec3 surface_reflection_ratio(in vec3 F_0, in vec3 eye_dir, in vec3 half_dir) {
    // Fresnel Schlick
    float term = 1.0 - dot(eye_dir, half_dir);
    return F_0 + (1 - F_0) * (term * term * term * term * term);
}

float geometric_occlusion_G(in vec3 v, in vec3 half_dir, in float k) {
    return dot(v, half_dir) / (dot(v, half_dir) * (1.0 - k) + k);
}

float geometric_occlusion(in float roughness, in vec3 light_dir, in vec3 half_dir, in vec3 normal) {
    const float coefficient = sqrt(2.0 / PI);
    float k = roughness * coefficient;
    return geometric_occlusion_G(light_dir, half_dir, k)
        * geometric_occlusion_G(normal, half_dir, k);
}

float microfaced_distribution(in float alpha, in vec3 half_dir, in vec3 normal) {
    float alpha_squared = alpha * alpha;
    float dot_prod = dot(normal, half_dir);
    float term = (dot_prod * dot_prod * (alpha_squared - 1.0) + 1.0);
    return alpha_squared / (PI * term * term);
}

vec3 brdf(in vec3 c_diff,
          in vec3 F_0,
          in float alpha,
          in float roughness,
          in vec3 eye_dir, // pointing outwards from the eye
          in vec3 light_dir, // pointing outwards from the light
          in vec3 half_dir,
          in vec3 normal) {
    vec3 F = surface_reflection_ratio(F_0, eye_dir, half_dir);
    float G = geometric_occlusion(roughness, light_dir, half_dir, normal);
    float D = microfaced_distribution(alpha, half_dir, normal);
    vec3 diffuse = c_diff / PI;

    vec3 f_diffuse = (1.0 - F) * diffuse;
    vec3 f_specular = (F * G * D)
        / (4.0 * dot(normal, light_dir) * dot(normal, eye_dir));

    return f_diffuse + f_specular;
}

vec3 pbr(in vec4 base_color,
         in float metallic,
         in float roughness,
         in vec3 eye_dir, // pointing outwards from the eye
         in vec3 light_dir, // pointing outwards from the light
         in vec3 normal) {
    const vec3 dielectric_specular = vec3(0.04, 0.04, 0.04);
    const vec3 black = vec3(0, 0, 0);

    vec3 half_dir = normalize(eye_dir + light_dir);
    vec3 c_diff = LERP(base_color.rgb * (1.0 - dielectric_specular.r), black, metallic);
    vec3 F_0 = LERP(dielectric_specular, base_color.rgb, metallic);
    float alpha = roughness * roughness;

    return brdf(c_diff,
                F_0,
                alpha,
                roughness,
                eye_dir,
                light_dir,
                half_dir,
                normal);
}

vec4 get_final_color(
        in vec2 dimensions,
        in mat4 view,
        in vec3 position,
        in vec3 normal,
        in vec4 tangent,
        in vec2 f_tex_coord,

        in bool base_color_texture_provided,
        in vec4 base_color_factor,
        in texture2D base_color_texture,
        in sampler base_color_sampler,

        in bool metallic_roughness_texture_provided,
        in vec2 metallic_roughness_factor,
        in texture2D metallic_roughness_texture,
        in sampler metallic_roughness_sampler,

        in bool normal_texture_provided,
        in float normal_texture_scale,
        in texture2D normal_texture,
        in sampler normal_sampler,

        in bool occlusion_texture_provided,
        in float occlusion_strength,
        in texture2D occlusion_texture,
        in sampler occlusion_sampler,

        in bool emissive_texture_provided,
        in vec3 emissive_factor,
        in texture2D emissive_texture,
        in sampler emissive_sampler
) {
    vec2 normalized_frag_coord = get_normalized_frag_coord(dimensions);
    vec3 bitangent = cross(normal, tangent.xyz) * tangent.w;
    vec4 base_color = sample_base_color(
        base_color_texture_provided,
        base_color_factor,
        base_color_texture,
        base_color_sampler,
        f_tex_coord
    );
    vec2 metallic_roughness = sample_metallic_roughness(
        metallic_roughness_texture_provided,
        metallic_roughness_factor,
        metallic_roughness_texture,
        metallic_roughness_sampler,
        f_tex_coord
    );
    vec3 sampled_normal = sample_normal(
        normal_texture_provided,
        normal_texture_scale,
        normal_texture,
        normal_sampler,
        f_tex_coord
    );
    float occlusion = sample_occlusion(
        occlusion_texture_provided,
        occlusion_strength,
        occlusion_texture,
        occlusion_sampler,
        f_tex_coord
    );
    vec3 emissive = sample_emissive(
        emissive_texture_provided,
        emissive_factor,
        emissive_texture,
        emissive_sampler,
        f_tex_coord
    );

    // Construct an orthonormal TBN matrix
    mat3 tangent_to_canonical = mat3(tangent.xyz, bitangent, normal);
    // We can use `transpose` to invert the matrix as it's orthonormal
    mat3 canonical_to_tangent = transpose(tangent_to_canonical);

    vec3 light_dir = vec3(0.0, 1.0, 0.0);
    float weight = clamp(dot(normal, light_dir), 0.0, 1.0);
    vec4 shaded = base_color * weight;
    vec4 result;

    /* result = vec4(0.0, metallic_roughness, 1.0); */

    if (normalized_frag_coord.x > 1.0 / 3.0) {
        /* normal = normal_sample.x * tangent.xyz + normal_sample.y * bitangent + */
        /*     normal_sample.z * normal; */
        /* result = vec4(vec3(occlusion), 1.0); */
        normal = tangent_to_canonical * sampled_normal;
        /* normal = normalize(normal); */
    }

    /* if (position.x < 0.3) { */
    /*     result = vec4(normal, 1.0); */
    /*     /1* result = vec4(dot(normal, normal), dot(tangent, tangent), dot(bitangent, bitangent), 1.0); *1/ */
    /* } else { */
    /*     result = texture( */
    /*         sampler2D(normal_texture, normal_sampler), */
    /*         vec2(f_tex_coord) */
    /*     ); */
    /* } */

    if (normalized_frag_coord.x > 2.0 / 3.0) {
        /* result = vec4(emissive, 1.0); */
        normal = normalize(normal);
    }

    /* result = base_color; */
    /* result = base_color * clamp(dot(normal, light_dir), 0.0, 1.0); */
    /* vec4 homogeneous_normal = inverse(transpose(view)) * vec4(normal, 0.0); */
    /* vec3 world_normal = PROJECT(homogeneous_normal); */
    /* result = vec4(world_normal, 1.0); */

    /* vec4 view_normal = transpose(inverse(view)) * transpose(inverse(model)) * transpose(inverse(matrix)) * vec4(normalized_normal, 0.0); */

    vec4 homogeneous_light_dir = view * vec4(light_dir, 0.0);
    vec3 view_light_dir = PROJECT(homogeneous_light_dir);
    /* vec4 homogeneous_normal = transpose(view) * vec4(normal, 0.0); */
    /* vec3 world_normal = PROJECT(homogeneous_normal); */
    float color_weight = clamp(dot(normal, view_light_dir), 0.0, 1.0);
    /* float color_weight = clamp(dot(world_normal, light_dir), 0.0, 1.0); */
    /* result = vec4(base_color.rgb * color_weight, base_color.a); */
    result = vec4(pbr(base_color,
                      metallic_roughness.x,
                      metallic_roughness.y,
                      vec3(0.0, 0.0, 1.0),
                      view_light_dir,
                      normal), 1.0);

    return result;
}
