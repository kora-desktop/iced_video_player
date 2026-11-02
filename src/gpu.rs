//! The GPU the renderer draws on, as far as a decoded frame cares: which
//! DRM modifiers its Vulkan device can copy from, and which render node it is.
//!
//! A DMA-BUF is laid out in its producer's tiling. On a laptop with two GPUs,
//! GStreamer's first VA decoder is often on the discrete one while the window
//! draws on the integrated one; the integrated GPU then reads the discrete
//! GPU's tiling as its own and draws noise. Each frame's modifier is checked
//! against the renderer's list before it is imported.

use std::collections::HashMap;
use std::sync::OnceLock;

use ash::vk;
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::hal::vulkan as vk_hal;

use crate::video::VideoFormat;

/// The renderer's render node, once a video has been drawn; see [`render_node`].
static RENDER_NODE: OnceLock<Option<u64>> = OnceLock::new();

/// The render node (as a `dev_t`) of the GPU videos are drawn on, once known.
pub(crate) fn render_node() -> Option<u64> {
    RENDER_NODE.get().copied().flatten()
}

/// What the renderer's device can take from a DMA-BUF.
#[derive(Debug, Default)]
pub(crate) struct RenderGpu {
    /// Modifiers the device copies from, per plane format.
    modifiers: HashMap<vk::Format, Vec<u64>>,
}

impl RenderGpu {
    /// Ask `device`. Not Vulkan, or no modifier support: takes nothing.
    pub(crate) fn of(device: &wgpu::Device) -> Self {
        // SAFETY: the HAL device is only read, and only while the guard lives.
        let Some(hal) = (unsafe { device.as_hal::<vk_hal::Api>() }) else {
            return Self::default();
        };
        let instance = hal.shared_instance().raw_instance();
        let physical = hal.raw_physical_device();
        let extensions = hal.enabled_device_extensions();
        let _ = RENDER_NODE.set(render_node_of(instance, physical, extensions));
        if !extensions.contains(&ash::ext::image_drm_format_modifier::NAME) {
            return Self::default();
        }
        let modifiers = [
            vk::Format::R8_UNORM,
            vk::Format::R8G8_UNORM,
            vk::Format::R16_UNORM,
            vk::Format::R16G16_UNORM,
        ]
        .into_iter()
        .map(|format| (format, copyable_modifiers(instance, physical, format)))
        .collect();
        Self { modifiers }
    }

    /// Whether both planes of a `format` frame laid out with `modifier` can
    /// be copied out of an imported image.
    pub(crate) fn reads(&self, format: VideoFormat, modifier: u64) -> bool {
        let planes = match format {
            VideoFormat::Nv12 => [vk::Format::R8_UNORM, vk::Format::R8G8_UNORM],
            VideoFormat::P010 => [vk::Format::R16_UNORM, vk::Format::R16G16_UNORM],
        };
        planes.iter().all(|plane| {
            self.modifiers
                .get(plane)
                .is_some_and(|list| list.contains(&modifier))
        })
    }
}

/// The modifiers `format` can be imported with and copied from.
fn copyable_modifiers(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    format: vk::Format,
) -> Vec<u64> {
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut properties = vk::FormatProperties2::default().push_next(&mut list);
    // SAFETY: the extension is enabled on this device; the first call only counts.
    unsafe {
        instance.get_physical_device_format_properties2(physical, format, &mut properties);
    }
    let count = list.drm_format_modifier_count as usize;
    if count == 0 {
        return Vec::new();
    }
    let mut entries = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
        .drm_format_modifier_properties(&mut entries);
    let mut properties = vk::FormatProperties2::default().push_next(&mut list);
    // SAFETY: `entries` holds as many as the first call counted.
    unsafe {
        instance.get_physical_device_format_properties2(physical, format, &mut properties);
    }
    let filled = list.drm_format_modifier_count as usize;
    entries
        .iter()
        .take(filled)
        .filter(|entry| {
            entry
                .drm_format_modifier_tiling_features
                .contains(vk::FormatFeatureFlags::TRANSFER_SRC)
        })
        .map(|entry| entry.drm_format_modifier)
        .collect()
}

/// The device's render node from `VK_EXT_physical_device_drm`, if it has one
/// and the driver supports the extension.
fn render_node_of(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    enabled: &[&std::ffi::CStr],
) -> Option<u64> {
    let name = ash::ext::physical_device_drm::NAME;
    let supported = enabled.contains(&name)
        // SAFETY: a plain query of the physical device's extensions.
        || unsafe { instance.enumerate_device_extension_properties(physical) }
            .unwrap_or_default()
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(name));
    if !supported {
        return None;
    }
    let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
    // SAFETY: the extension is supported, so its struct may be chained.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    if drm.has_render == vk::FALSE {
        return None;
    }
    Some(libc::makedev(
        u32::try_from(drm.render_major).ok()?,
        u32::try_from(drm.render_minor).ok()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An AMD GFX9 integrated GPU's linear and tiled NV12 plane modifiers.
    const LINEAR: u64 = 0;
    const GFX9_TILED: u64 = 0x0200_0000_0000_0d01;
    /// A tiling only an RDNA2 discrete GPU uses.
    const GFX10_RBPLUS: u64 = 0x0200_0000_0040_0d01;

    fn gpu(modifiers: &[u64]) -> RenderGpu {
        RenderGpu {
            modifiers: [vk::Format::R8_UNORM, vk::Format::R8G8_UNORM]
                .into_iter()
                .map(|f| (f, modifiers.to_vec()))
                .collect(),
        }
    }

    #[test]
    fn another_gpus_tiling_is_not_read() {
        let integrated = gpu(&[LINEAR, GFX9_TILED]);
        assert!(!integrated.reads(VideoFormat::Nv12, GFX10_RBPLUS));
    }

    #[test]
    fn its_own_tiling_and_linear_are_read() {
        let integrated = gpu(&[LINEAR, GFX9_TILED]);
        assert!(integrated.reads(VideoFormat::Nv12, GFX9_TILED));
        assert!(integrated.reads(VideoFormat::Nv12, LINEAR));
    }

    #[test]
    fn both_planes_must_take_it() {
        let mut gpu = gpu(&[LINEAR, GFX9_TILED]);
        gpu.modifiers.insert(vk::Format::R8G8_UNORM, vec![LINEAR]);
        assert!(!gpu.reads(VideoFormat::Nv12, GFX9_TILED));
    }

    #[test]
    fn a_device_that_lists_nothing_takes_nothing() {
        assert!(!RenderGpu::default().reads(VideoFormat::Nv12, LINEAR));
        assert!(!gpu(&[LINEAR]).reads(VideoFormat::P010, LINEAR));
    }
}
