use std;
use std::collections::HashSet;
use std::time::Duration;
use winit::{MouseButton, VirtualKeyCode};
use boolinator::Boolinator;
use ammolite_math::*;

pub trait Camera: std::fmt::Debug {
    /**
     * Returns the view matrix intended to be used in the MVP matrix chain.
     */
    fn get_view_matrix(&self) -> Mat4;

    /**
     * Returns the position of the camera in the world.
     */
    fn get_position(&self) -> Vec3;

    /**
     * Returns the forward unit vector of the camera.
     */
    fn get_direction(&self) -> Vec3;

    fn get_rotation_axis_angles(&self) -> Vec3;

    /**
     * Handle controls.
     */
    fn update(&mut self,
              delta_time: &Duration,
              cursor_delta: &[f64; 2],
              pressed_keys: &HashSet<VirtualKeyCode>,
              pressed_mouse_buttons: &HashSet<MouseButton>);
}

#[derive(Debug, Clone, PartialEq)]
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
            yaw: std::f64::consts::PI,
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
        * Mat4::translation(&-self.get_position())
    }

    fn get_position(&self) -> Vec3 {
        self.position.clone()
    }

    fn get_direction(&self) -> Vec3 {
        [
            -(self.pitch.cos() * self.yaw.sin()) as f32,
            self.pitch.sin() as f32,
            -(self.pitch.cos() * self.yaw.cos()) as f32,
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

        let forward: Vec3 = (rotation_matrix * Vec3::from([0.0, 0.0, -1.0]).into_homogeneous()).into_projected();
        let right = forward.cross(&Vec3::from([0.0, 1.0, 0.0]));
        let mut direction: Vec3 = Vec3::zero();

        pressed_keys.contains(&VirtualKeyCode::W).as_option()
            .map(|()| direction += &forward);
        pressed_keys.contains(&VirtualKeyCode::S).as_option()
            .map(|()| direction -= &forward);
        pressed_keys.contains(&VirtualKeyCode::A).as_option()
            .map(|()| direction -= &right);
        pressed_keys.contains(&VirtualKeyCode::D).as_option()
            .map(|()| direction += &right);
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

    fn get_rotation_axis_angles(&self) -> Vec3 {
        [self.pitch as f32, self.yaw as f32, 0.0].into()
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

// pub fn construct_perspective_projection_matrix(near_plane: f32, far_plane: f32, aspect_ratio: f32, fov_rad: f32) -> Mat4 {
//     // This function generates matrix that transforms the view frustum to normalized device
//     // coordinates, which is, in case of Vulkan, the set [-1; 1]x[-1; 1]x[0; 1].

//     // The resulting `(x, y, z, w)` vector gets normalized following the execution of the vertex
//     // shader to `(x/w, y/w, z/w)` (W-division). This makes it possible to create a perspective
//     // projection matrix.
//     // We copy the Z coordinate to the W coordinate so as to divide all coordinates by Z.
//     let z_n = near_plane;
//     let z_f = far_plane;
//     // With normalization, it is actually `1 / (z * tan(FOV / 2))`, which is the width of the
//     // screen at that point in space of the vector.
//     // The X coordinate needs to be divided by the aspect ratio to make it independent of the
//     // window size.
//     // Even though we could negate the Y coordinate so as to adjust the vectors to the Vulkan
//     // coordinate system, which has the Y axis pointing downwards, contrary to OpenGL, we need to
//     // apply the same transformation to other vertex attributes such as normal and tangent vectors,
//     // but those are not projected.
//     let f = 1.0 / (fov_rad / 2.0).tan();

//     // Because we know, from the nature of the perspective projection, that the
//     // Z coordinate must be scaled and translated, we will be deriving A, B from
//     // the equation:
//     // `f(z) = A*z + B`,
//     // where A and B are the 3rd, respectively 4th column coefficients of the 3rd row.
//     // The equation changes to the following, after the W-division:
//     // `f(z) = A + B/z`
//     // And must satisfy the following conditions:
//     // `f(z_n) = 0`
//     // `f(z_f) = 1`
//     // Solving for A and B gives us the necessary coefficients to construct the matrix:
//     // A = (z_n * z_f) / (z_n - z_f)
//     // B = -z_f / (z_n - z_f)
//     // mat4!([f / aspect_ratio, 0.0,                0.0,                       0.0,
//     //                     0.0,  -f,                0.0,                       0.0,
//     //                     0.0, 0.0, -z_f / (z_n - z_f), (z_n * z_f) / (z_n - z_f),
//     //                     0.0, 0.0,                1.0,                       0.0])

//     // TODO: Investigate the mysterious requirement of flipping the X coordinate
//     // mat4!([-f / aspect_ratio, 0.0,                0.0,                       0.0,
//     //                      0.0,   f,                0.0,                       0.0,
//     //                      0.0, 0.0, -z_f / (z_n - z_f), (z_n * z_f) / (z_n - z_f),
//     //                      0.0, 0.0,                1.0,                       0.0])

//     // glTF spec formula:
//     mat4!([f / aspect_ratio, 0.0,     0.0,  0.0,
//            0.0,              1.0 * f, 0.0,  0.0,
//            0.0,              0.0,     -1.0, -2.0 * z_n,
//            0.0,              0.0,     -1.0, 0.0])
// }

pub fn construct_perspective_projection_matrix_asymmetric(near_plane: f32, far_plane: f32, angle_right: f32, angle_up: f32, angle_left: f32, angle_down: f32) -> Mat4 {
    // See http://www.songho.ca/opengl/gl_projectionmatrix.html
    // for a brilliant derivation of the projection matrix.

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

    let l = angle_left.tan() * z_n;
    let r = angle_right.tan() * z_n;
    let b = angle_down.tan() * z_n;
    let t = angle_up.tan() * z_n;

    // mat4!([-2.0 * idx,       0.0, sx * idx, 0.0,
    //              0.0, 2.0 * idy, sy * idy, 0.0,
    //              0.0,       0.0,      0.0, z_n,
    //              0.0,       0.0,     -1.0, 0.0])
    mat4!([(2.0 * z_n) / (r - l),                   0.0,  (r + l) / (r - l),                       0.0,
                             0.0, (2.0 * z_n) / (t - b),  (t + b) / (t - b),                       0.0,
                             0.0,                   0.0,  z_f / (z_n - z_f), (z_n * z_f) / (z_n - z_f),
                             0.0,                   0.0,               -1.0,                       0.0])
}

// pub fn construct_perspective_projection_matrix(near_plane: f32, far_plane: f32, aspect_ratio: f32, fov_rad: f32) -> Mat4 {
//     let angle_up = fov_rad / aspect_ratio;
//     let angle_right = fov_rad;
//     let angle_down = -angle_up;
//     let angle_left = -angle_right;

//     construct_perspective_projection_matrix_asymmetric(
//         near_plane, far_plane,
//         angle_right, angle_up, angle_left, angle_down,
//     )
// }
