#include "gltf_common.h"

#define GET_FINAL_COLOR() get_final_color(      \
        dimensions,                             \
        view,                                   \
        camera_position,                        \
        f_world_position,                       \
        f_world_normal,                         \
        f_world_tangent,                        \
        f_tex_coord,                            \
                                                \
        base_color_texture_provided,            \
        base_color_factor,                      \
        base_color_texture,                     \
        base_color_sampler,                     \
                                                \
        metallic_roughness_texture_provided,    \
        metallic_roughness_factor,              \
        metallic_roughness_texture,             \
        metallic_roughness_sampler,             \
                                                \
        normal_texture_provided,                \
        normal_texture_scale,                   \
        normal_texture,                         \
        normal_sampler,                         \
                                                \
        occlusion_texture_provided,             \
        occlusion_strength,                     \
        occlusion_texture,                      \
        occlusion_sampler,                      \
                                                \
        emissive_texture_provided,              \
        emissive_factor,                        \
        emissive_texture,                       \
        emissive_sampler                        \
    )

struct BRDFParams {
    float NdotL;
    float NdotV;
    float NdotH;
    float LdotH;
    float VdotH;
    vec3 c_diff;
    vec3 F_0;
    float alpha;
    float roughness;
};

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
    const float min_roughness = 0.04;
    vec2 metallic_roughness;

    if (metallic_roughness_texture_provided) {
        vec4 texture_value = texture(
            sampler2D(metallic_roughness_texture, metallic_roughness_sampler),
            vec2(f_tex_coord)
        );

        // components stored in the opposite order
        metallic_roughness = metallic_roughness_factor * texture_value.bg;
    } else {
        metallic_roughness = metallic_roughness_factor;
    }

    return vec2(metallic_roughness.x, clamp(metallic_roughness.y, min_roughness, 1.0));
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

vec3 surface_reflection_ratio(in BRDFParams params) {
    /* float reflectance = max(max(params.F_0.r, params.F_0.g), params.F_0.b); */
    /* float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0); */
    /* vec3 r0 = params.F_0; */
    /* vec3 r90 = vec3(1.0, 1.0, 1.0) * reflectance90; */

    /* return r0 + (r90 - r0) * pow(clamp(1.0 - params.VdotH, 0.0, 1.0), 5.0); */

    // Fresnel Schlick
    float term = 1.0 - params.VdotH;
    return params.F_0 + (1 - params.F_0) * (term * term * term * term * term);
}

float geometric_occlusion(in BRDFParams params) {
    /* return min( */
    /*            min( */
    /*                2.0 * params.NdotV * params.NdotH / params.VdotH, */
    /*                2.0 * params.NdotL * params.NdotH / params.VdotH), */
    /*            1.0 */
    /*        ); */
    const float coefficient = sqrt(2.0 / PI);
    float k = params.roughness * coefficient;
    float G_L = params.LdotH / (params.LdotH * (1.0 - k) + k);
    float G_N = params.NdotH / (params.NdotH * (1.0 - k) + k);
    return G_L * G_N;
}

float microfaced_distribution(in BRDFParams params) {
    float alpha_squared = params.alpha * params.alpha;
    float term = (params.NdotH * params.NdotH * (alpha_squared - 1.0) + 1.0);
    return alpha_squared / (PI * term * term);
}

// See https://github.com/KhronosGroup/glTF-WebGL-PBR/ for an exemplary
// implementation
vec3 brdf(in BRDFParams params, out vec3 F, out float G, out float D) {
    F = surface_reflection_ratio(params);
    G = geometric_occlusion(params);
    D = microfaced_distribution(params);
    vec3 diffuse = params.c_diff / PI;

    vec3 f_diffuse = (1.0 - F) * diffuse;
    vec3 f_specular = (F * G * D)
        / (4.0 * params.NdotL * params.NdotV);

    return f_diffuse + f_specular;
}

vec3 pbr(in vec4 base_color,
         in float metallic,
         in float roughness,
         in vec3 eye_dir, // pointing outwards from the eye
         in vec3 light_dir, // pointing outwards from the light
         in vec3 normal,
         out vec3 F,
         out float G,
         out float D) {
    const vec3 dielectric_specular = vec3(0.04, 0.04, 0.04);
    const vec3 black = vec3(0, 0, 0);

    vec3 half_dir = normalize(eye_dir + light_dir);
    vec3 c_diff = mix(base_color.rgb * (1.0 - dielectric_specular.r), black, metallic);
    vec3 F_0 = mix(dielectric_specular, base_color.rgb, metallic);
    float alpha = roughness * roughness;

    vec3 V = eye_dir;
    vec3 L = light_dir;
    vec3 H = half_dir;
    vec3 N = normal;

    BRDFParams brdf_params = BRDFParams(
        clamp(dot(N, L), 0.001, 1.0),
        clamp(abs(dot(N, V)), 0.001, 1.0),
        clamp(dot(N, H), 0.0, 1.0),
        clamp(dot(L, H), 0.0, 1.0),
        clamp(dot(V, H), 0.0, 1.0),
        c_diff,
        F_0,
        alpha,
        roughness
    );

    return brdf(brdf_params, F, G, D);
}

// Immediately returns if the current fragment is within the specified region.
#define VISUALIZE_VECTOR_INVERT(vector, top_left, bottom_right, dimensions) do {  \
    vec2 coord = get_normalized_frag_coord(dimensions);                           \
                                                                                  \
    if (coord.x > top_left.x && coord.y > top_left.y                              \
            && coord.x < bottom_right.x && coord.y < bottom_right.y) {            \
        vec2 coord_within_region = (coord - top_left)                             \
            / (bottom_right - top_left);                                          \
                                                                                  \
        if ((coord_within_region.x + coord_within_region.y > 1.0)                 \
                != (coord_within_region.x - coord_within_region.y + 1.0 < 1.0)) { \
            vector *= -1.0;                                                       \
        }                                                                         \
                                                                                  \
        return vec4(vector, 1.0);                                                 \
    }                                                                             \
} while (false);

#define VISUALIZE_VECTOR(vector, top_left, bottom_right, dimensions) do {      \
    vec2 coord = get_normalized_frag_coord(dimensions);                        \
                                                                               \
    if (coord.x > top_left.x && coord.y > top_left.y                           \
            && coord.x < bottom_right.x && coord.y < bottom_right.y) {         \
        return vec4(vector, 1.0);                                              \
    }                                                                          \
} while (false);

vec4 get_final_color(
        in vec2 dimensions,
        in mat4 view,
        in vec3 camera_position,
        in vec3 world_position,
        in vec3 world_normal,
        in vec4 world_tangent,
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
    vec3 world_bitangent = cross(world_normal, world_tangent.xyz) * world_tangent.w;
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
    mat3 tangent_to_canonical = mat3(world_tangent.xyz, world_bitangent, world_normal);
    // We can use `transpose` to invert the matrix as it's orthonormal
    mat3 canonical_to_tangent = transpose(tangent_to_canonical);

    vec3 light_world_position[3] = {
        vec3(1.0, 1.5, 2.0),
        vec3(-1.0, -1.5, 2.0),
        vec3(-1.0, 1.5, -2.0)
    };
    world_normal = normalize(tangent_to_canonical * sampled_normal);

    /* float color_weight = clamp(dot(world_normal, light_world_direction), 0.0, 1.0); */
    /* vec3 diffuse_color = base_color.rgb * mix(0.1, 1.0, color_weight); */
    /* return vec4(diffuse_color, base_color.a); */

    vec3 eye_direction = normalize(world_position - camera_position);
    vec3 accumulated_color = vec3(0.0);

    for (int i = 0; i < 3; i++) {
        vec3 light_world_direction = normalize(world_position - light_world_position[i]);
        float HdotV = dot(light_world_direction, eye_direction);
        vec3 vecHdotV = vec3(HdotV);

        /* VISUALIZE_VECTOR_INVERT(world_normal, vec2(0.0, 0.0), vec2(1.0, 0.5), dimensions); */
        /* VISUALIZE_VECTOR_INVERT(light_world_direction, vec2(0.0, 0.5), vec2(1.0, 1.0), dimensions); */
        /* VISUALIZE_VECTOR(vecHdotV, vec2(0.0, 0.0), vec2(1.0, 1.0), dimensions); */

        /* VISUALIZE_VECTOR_INVERT(eye_direction, vec2(0.0 / 3.0, 0.5), vec2(1.0 / 3.0, 1.0), dimensions); */
        /* VISUALIZE_VECTOR_INVERT(light_world_direction, vec2(1.0 / 3.0, 0.5), vec2(2.0 / 3.0, 1.0), dimensions); */
        /* VISUALIZE_VECTOR_INVERT(world_normal, vec2(2.0 / 3.0, 0.5), vec2(3.0 / 3.0, 1.0), dimensions); */

        vec3 F;
        float G;
        float D;
        vec3 f = pbr(base_color,
                     metallic_roughness.x,
                     metallic_roughness.y,
                     eye_direction,
                     light_world_direction,
                     world_normal,
                     F, G, D);

        accumulated_color += f;

        /* VISUALIZE_VECTOR(base_color.rgb, vec2(0.5, 0.0), vec2(1.0, 0.5), dimensions); */
        /* VISUALIZE_VECTOR(metallic_roughness.xxx, vec2(0.0, 0.5), vec2(0.5, 1.0), dimensions); */
        /* VISUALIZE_VECTOR(metallic_roughness.yyy, vec2(0.5, 0.5), vec2(1.0, 1.0), dimensions); */

        /* VISUALIZE_VECTOR(vec3(D), vec2(0.0), vec2(1.0), dimensions); */

        // TODO: Transform all primitives to World space, where possible.
        /* VISUALIZE_VECTOR(F, vec2(0.0 / 3.0, 0.5), vec2(1.0 / 3.0, 1.0), dimensions); */
        /* VISUALIZE_VECTOR(vec3(G), vec2(1.0 / 3.0, 0.5), vec2(2.0 / 3.0, 1.0), dimensions); */
        /* VISUALIZE_VECTOR(vec3(D), vec2(2.0 / 3.0, 0.5), vec2(3.0 / 3.0, 1.0), dimensions); */

        /* VISUALIZE_VECTOR(vec3(D), vec2(0.0, 0.0), vec2(1.0, 1.0), dimensions); */
    }

    return vec4(accumulated_color / 3.0, base_color.a);
}
