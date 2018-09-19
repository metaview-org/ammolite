use vulkano::sampler::SamplerAddressMode;
use vulkano::sampler::Filter;
use vulkano::sampler::MipmapMode;
use gltf::texture::WrappingMode;
use gltf::texture::MagFilter;
use gltf::texture::MinFilter;

pub trait IntoVulkanEquivalent {
    type Output;

    fn into_vulkan_equivalent(self) -> Self::Output;
}

impl IntoVulkanEquivalent for WrappingMode {
    type Output = SamplerAddressMode;

    fn into_vulkan_equivalent(self) -> Self::Output {
        match self {
            WrappingMode::ClampToEdge => SamplerAddressMode::ClampToEdge,
            WrappingMode::MirroredRepeat => SamplerAddressMode::MirroredRepeat,
            WrappingMode::Repeat => SamplerAddressMode::Repeat,
        }
    }
}

impl IntoVulkanEquivalent for MagFilter {
    type Output = Filter;

    fn into_vulkan_equivalent(self) -> Self::Output {
        match self {
            MagFilter::Nearest => Filter::Nearest,
            MagFilter::Linear => Filter::Linear,
        }
    }
}

impl IntoVulkanEquivalent for MinFilter {
    type Output = (Filter, MipmapMode);

    fn into_vulkan_equivalent(self) -> Self::Output {
        match self {
            MinFilter::Nearest => (Filter::Nearest, MipmapMode::Linear),
            MinFilter::Linear => (Filter::Linear, MipmapMode::Linear),
            MinFilter::NearestMipmapNearest => (Filter::Nearest, MipmapMode::Nearest),
            MinFilter::LinearMipmapNearest => (Filter::Linear, MipmapMode::Nearest),
            MinFilter::NearestMipmapLinear => (Filter::Nearest, MipmapMode::Linear),
            MinFilter::LinearMipmapLinear => (Filter::Linear, MipmapMode::Linear),
        }
    }
}
