#[derive(Debug, Fail)]
pub enum ModelInitializationError {
    #[fail(display = "The model has already been initialized and cannot be initialized again")]
    ModelAlreadyInitialized,
}

#[derive(Debug, Fail)]
pub enum ModelDrawError {
    #[fail(display = "Trying to draw the default scene while no default scene is specified")]
    NoDefaultScene,
    #[fail(display = "Invalid toolchain index: {}", index)]
    InvalidSceneIndex {
        index: usize,
    },
}
