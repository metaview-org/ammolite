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
              cursor_delta: &(f64, f64),
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

    fn update_rotation(&mut self, cursor_delta: &(f64, f64)) {
        let (delta_yaw, delta_pitch) = cursor_delta;

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
              cursor_delta: &(f64, f64),
              pressed_keys: &HashSet<VirtualKeyCode>,
              pressed_mouse_buttons: &HashSet<MouseButton>) {
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
