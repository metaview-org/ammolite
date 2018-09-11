#[derive(Debug, Fail)]
pub enum DrawError {
    #[fail(display = "Trying to draw the default scene while no default scene is specified")]
    NoDefaultScene,
    #[fail(display = "Invalid toolchain index: {}", index)]
    InvalidSceneIndex {
        index: usize,
    },
}
