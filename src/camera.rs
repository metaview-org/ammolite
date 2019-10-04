use std;
use std::collections::HashSet;
use std::time::Duration;
use winit::{MouseButton, VirtualKeyCode};
use boolinator::Boolinator;
use crate::math::*;

pub trait Camera {
    fn get_view_matrix(&self) -> Mat4;
    fn get_position(&self) -> Vec3;
    fn get_direction(&self) -> Vec3;
    fn update(&mut self,
              delta_time: &Duration,
              cursor_delta: &[f64; 2],
              pressed_keys: &HashSet<VirtualKeyCode>,
              pressed_mouse_buttons: &HashSet<MouseButton>);
}

#[derive(Clone, PartialEq)]
pub struct PitchYawCamera3 {
    position: Vec3,
    yaw: f64,
    pitch: f64,
    mouse_sensitivity: f64,
    speed: f64,
    fov: f64,
}

unsafe impl Sync for PitchYawCamera3 {}

impl PitchYawCamera3 {
    pub fn new() -> PitchYawCamera3 {
        Self {
            position: [0.0, 0.0, 0.0].into(),
            yaw: 0.0,
            pitch: 0.0,
            mouse_sensitivity: 0.002,
            speed: 4.0,
            fov: std::f64::consts::FRAC_PI_2,
        }
    }

    pub fn new_with_position(position: Vec3) -> Self {
        Self {
            position,
            .. Self::default()
        }
    }

    fn update_rotation(&mut self, cursor_delta: &[f64; 2]) {
        let [delta_yaw, delta_pitch] = cursor_delta;

        self.yaw -= delta_yaw * self.mouse_sensitivity;
        self.pitch -= delta_pitch * self.mouse_sensitivity;

        if self.pitch > std::f64::consts::FRAC_PI_2 {
            self.pitch = std::f64::consts::FRAC_PI_2;
        } else if self.pitch < -std::f64::consts::FRAC_PI_2 {
            self.pitch = -std::f64::consts::FRAC_PI_2;
        }
    }

    fn get_translation_vector(&self) -> Vec3 {
        self.position.clone()
    }

    fn get_rotation_axis_angles(&self) -> Vec3 {
        [-self.pitch as f32, self.yaw as f32, 0.0].into()
    }
}

impl Default for PitchYawCamera3 {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera for PitchYawCamera3 {
    fn get_view_matrix(&self) -> Mat4 {
        let rotation = -self.get_rotation_axis_angles();

        Mat4::rotation_pitch(rotation[0])
        * Mat4::rotation_yaw(rotation[1])
        * Mat4::rotation_roll(rotation[2])
        * Mat4::translation(&-self.get_translation_vector())
    }

    fn get_position(&self) -> Vec3 {
        self.position.clone()
    }

    fn get_direction(&self) -> Vec3 {
        [
            (self.pitch.cos() * self.yaw.cos()) as f32,
            self.pitch.sin() as f32,
            (self.pitch.cos() * self.yaw.sin()) as f32,
        ].into()
    }

    fn update(&mut self,
              delta_time: &Duration,
              cursor_delta: &[f64; 2],
              pressed_keys: &HashSet<VirtualKeyCode>,
              _pressed_mouse_buttons: &HashSet<MouseButton>) {
        self.update_rotation(cursor_delta);

        let delta_seconds = (delta_time.as_nanos() as f64) / 1.0e9;
        let distance = self.speed * delta_seconds;

        if distance == 0.0 {
            return;
        }

        let rotation = self.get_rotation_axis_angles();
        let rotation_matrix = Mat4::rotation_roll(rotation[2])
            * Mat4::rotation_yaw(rotation[1])
            * Mat4::rotation_pitch(rotation[0]);

        let forward: Vec3 = (rotation_matrix * Vec3::from([0.0, 0.0, 1.0]).into_homogeneous()).into_projected();
        // let forward = Vec3::from([0.0, 0.0, 1.0]);
        let left = Vec3::from([0.0, 1.0, 0.0]).cross(&forward);
        let mut direction: Vec3 = Vec3::zero();

        pressed_keys.contains(&VirtualKeyCode::W).as_option()
            .map(|()| direction += &forward);
        pressed_keys.contains(&VirtualKeyCode::S).as_option()
            .map(|()| direction -= &forward);
        pressed_keys.contains(&VirtualKeyCode::A).as_option()
            .map(|()| direction += &left);
        pressed_keys.contains(&VirtualKeyCode::D).as_option()
            .map(|()| direction -= &left);
        pressed_keys.contains(&VirtualKeyCode::LShift).as_option()
            .map(|()| direction += Vec3::from([0.0, 1.0, 0.0]));
        pressed_keys.contains(&VirtualKeyCode::LControl).as_option()
            .map(|()| direction -= Vec3::from([0.0, 1.0, 0.0]));

        let direction_norm = direction.norm();

        if direction_norm != 0.0 {
            direction *= distance as f32 / direction_norm;

            pressed_keys.contains(&VirtualKeyCode::Space).as_option()
                .map(|()| direction *= 0.1);

            self.position += &direction;
        }
    }
}

pub fn construct_model_matrix(scale: f32, translation: &Vec3, rotation: &Vec3) -> Mat4 {
    Mat4::translation(translation)
        * Mat4::rotation_roll(rotation[2])
        * Mat4::rotation_yaw(rotation[1])
        * Mat4::rotation_pitch(rotation[0])
        * Mat4::scale(scale)
}

pub fn construct_view_matrix(translation: &Vec3, rotation: &Vec3) -> Mat4 {
    // construct_model_matrix(1.0, &-translation, &Vec3::zero())
    construct_model_matrix(1.0, &-translation, &-rotation)
}

pub fn construct_orthographic_projection_matrix(near_plane: f32, far_plane: f32, dimensions: Vec2) -> Mat4 {
    let z_n = near_plane;
    let z_f = far_plane;

    // Scale the X/Y-coordinates according to the dimensions. Translate and scale the Z-coordinate.
    mat4!([1.0 / dimensions[0],                 0.0,               0.0,                0.0,
                           0.0, 1.0 / dimensions[1],               0.0,                0.0,
                           0.0,                 0.0, 1.0 / (z_f - z_n), -z_n / (z_f - z_n),
                           0.0,                 0.0,               0.0,                1.0])
}

pub fn construct_perspective_projection_matrix(near_plane: f32, far_plane: f32, aspect_ratio: f32, fov_rad: f32) -> Mat4 {
    // The resulting `(x, y, z, w)` vector gets normalized following the execution of the vertex
    // shader to `(x/w, y/w, z/w)` (W-division). This makes it possible to create a perspective
    // projection matrix.
    // We copy the Z coordinate to the W coordinate so as to divide all coordinates by Z.
    let z_n = near_plane;
    let z_f = far_plane;
    // With normalization, it is actually `1 / (z * tan(FOV / 2))`, which is the width of the
    // screen at that point in space of the vector.
    // The X coordinate needs to be divided by the aspect ratio to make it independent of the
    // window size.
    // Even though we could negate the Y coordinate so as to adjust the vectors to the Vulkan
    // coordinate system, which has the Y axis pointing downwards, contrary to OpenGL, we need to
    // apply the same transformation to other vertex attributes such as normal and tangent vectors,
    // but those are not projected.
    let f = 1.0 / (fov_rad / 2.0).tan();

    // We derive the coefficients for the Z coordinate from the following equation:
    // `f(z) = A*z + B`, because we know we need to translate and scale the Z coordinate.
    // The equation changes to the following, after the W-division:
    // `f(z) = A + B/z`
    // And must satisfy the following conditions:
    // `f(z_near) = 0`
    // `f(z_far) = 1`
    // Solving for A and B gives us the necessary coefficients to construct the matrix.
    // mat4!([f / aspect_ratio, 0.0,                0.0,                       0.0,
    //                     0.0,  -f,                0.0,                       0.0,
    //                     0.0, 0.0, -z_f / (z_n - z_f), (z_n * z_f) / (z_n - z_f),
    //                     0.0, 0.0,                1.0,                       0.0])

    // TODO: Investigate the mysterious requirement of flipping the X coordinate
    mat4!([-f / aspect_ratio, 0.0,                0.0,                       0.0,
                         0.0,   f,                0.0,                       0.0,
                         0.0, 0.0, -z_f / (z_n - z_f), (z_n * z_f) / (z_n - z_f),
                         0.0, 0.0,                1.0,                       0.0])

    // glTF spec formula:
    // mat4!([1.0 / (aspect_ratio * f), 0.0,     0.0,  0.0,
    //        0.0,                      1.0 * f, 0.0,  0.0,
    //        0.0,                      0.0,     -1.0, -2.0 * z_n,
    //        0.0,                      0.0,     -1.0, 0.0])
}
