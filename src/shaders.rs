pub mod gltf_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/gltf.vert",
    }
}

pub mod gltf_opaque_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/gltf_opaque.frag",
    }
}

pub mod gltf_mask_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/gltf_mask.frag",
    }
}

pub mod gltf_blend_preprocess_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/gltf_blend_preprocess.frag",
    }
}

pub mod gltf_blend_finalize_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/gltf_blend_finalize.frag",
    }
}

use crate::math::matrix::*;
use crate::math::vector::*;

pub use crate::shaders::gltf_opaque_frag::ty::*;

impl SceneUBO {
    pub fn new(time_elapsed: f32, dimensions: Vec2, camera_position: Vec3, model: Mat4, view: Mat4, projection: Mat4) -> SceneUBO {
        SceneUBO {
            time_elapsed,
            dimensions: dimensions.0,
            camera_position: camera_position.0,
            model: model.0,
            view: view.0,
            projection: projection.0,
            _dummy0: Default::default(),
            _dummy1: Default::default(),
        }
    }
}

impl NodeUBO {
    pub fn new(matrix: Mat4) -> NodeUBO {
        NodeUBO {
            matrix: matrix.0,
        }
    }
}

impl MaterialUBO {
    pub fn new(
        alpha_cutoff: f32,
        base_color_texture_provided: bool,
        base_color_factor: Vec4,
        metallic_roughness_texture_provided: bool,
        metallic_roughness_factor: Vec2,
        normal_texture_provided: bool,
        normal_texture_scale: f32,
        occlusion_texture_provided: bool,
        occlusion_strength: f32,
        emissive_texture_provided: bool,
        emissive_factor: Vec3,
    ) -> Self {
        MaterialUBO {
            alpha_cutoff,
            base_color_texture_provided: base_color_texture_provided as u32,
            base_color_factor: base_color_factor.0,
            metallic_roughness_texture_provided: metallic_roughness_texture_provided as u32,
            metallic_roughness_factor: metallic_roughness_factor.0,
            normal_texture_provided: normal_texture_provided as u32,
            normal_texture_scale,
            occlusion_texture_provided: occlusion_texture_provided as u32,
            occlusion_strength,
            emissive_texture_provided: emissive_texture_provided as u32,
            emissive_factor: emissive_factor.0,
            _dummy0: Default::default(),
            _dummy1: Default::default(),
            _dummy2: Default::default(),
        }
    }
}

impl Default for MaterialUBO {
    fn default() -> Self {
        Self::new(
            0.5,
            false,
            [1.0, 1.0, 1.0, 1.0].into(),
            false,
            [1.0, 1.0].into(),
            false,
            1.0,
            false,
            1.0,
            false,
            [0.0, 0.0, 0.0].into(),
        )
    }
}

impl PushConstants {
    pub fn new(vertex_color_provided: bool) -> Self {
        Self {
            vertex_color_provided: vertex_color_provided as u32,
        }
    }
}
