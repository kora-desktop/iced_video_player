//! VA-API DMA-BUF zero-copy verification (AMD and Intel).
//!
//! Decodes one frame twice: on the GPU through the VA decoder, handed over as a
//! DMA-BUF (`video/x-raw(memory:DMABuf),format=DMA_DRM`), and on the CPU
//! through `avdec_h264`. The DMA-BUF planes are imported into wgpu through the
//! Vulkan HAL with the DRM modifier the decoder reported, read back, and
//! compared byte for byte against the CPU decode — H.264 decoding is
//! bit-exact, so any difference is the import's.
//!
//! Usage:
//!   cargo run --example va_dmabuf_verify -- <video.mp4> [frame]

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_allocators::DmaBufMemory;
use gstreamer_app as gst_app;
use gstreamer_video::{VideoFrameExt, VideoMeta};
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::hal::vulkan as vk_hal;

const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "iced_video_player=info,wgpu_hal=warn".into()),
        )
        .init();
    gst::init().unwrap();

    let mut args = std::env::args().skip(1);
    let path = std::fs::canonicalize(
        args.next()
            .expect("usage: va_dmabuf_verify <video> [frame]"),
    )
    .expect("video not found");
    let nth: usize = args.next().map_or(10, |n| n.parse().expect("frame index"));
    let location = path.display().to_string().replace('"', "\\\"");

    let gpu = pull_nth(
        &format!(
            "filesrc location=\"{location}\" ! qtdemux ! h264parse ! vah264dec ! \
             appsink name=sink sync=false caps=video/x-raw(memory:DMABuf),format=DMA_DRM"
        ),
        nth,
    );
    let cpu = pull_nth(
        &format!(
            "filesrc location=\"{location}\" ! qtdemux ! h264parse ! avdec_h264 ! videoconvert ! \
             appsink name=sink sync=false caps=video/x-raw,format=NV12"
        ),
        nth,
    );

    let (gpu_sample, gpu_pipeline) = gpu;
    let (cpu_sample, cpu_pipeline) = cpu;
    let gpu_buffer = gpu_sample.buffer().unwrap();
    let cpu_buffer = cpu_sample.buffer().unwrap();
    assert_eq!(
        gpu_buffer.pts(),
        cpu_buffer.pts(),
        "the two decodes are on different frames"
    );

    let caps = gpu_sample.caps().unwrap();
    let s = caps.structure(0).unwrap();
    println!("GPU caps: {s}");
    let width = s.get::<i32>("width").unwrap() as u32;
    let height = s.get::<i32>("height").unwrap() as u32;
    let drm_format = s
        .get::<&str>("drm-format")
        .expect("no drm-format in DMA_DRM caps");
    let (fourcc, modifier) = drm_format.split_once(':').unwrap_or((drm_format, "0x0"));
    let modifier = u64::from_str_radix(modifier.trim_start_matches("0x"), 16).unwrap();
    assert_eq!(fourcc, "NV12", "this check compares NV12");
    println!("frame {nth}: {width}x{height} {fourcc} modifier=0x{modifier:016x}");

    println!("buffer: {} memories", gpu_buffer.n_memory());
    for i in 0..gpu_buffer.n_memory() {
        let mem = gpu_buffer.peek_memory(i);
        let dmabuf = mem.downcast_memory_ref::<DmaBufMemory>();
        println!(
            "  mem[{i}]: size={} offset={} dmabuf={} fd={:?} fd_size={:?}",
            mem.size(),
            mem.offset(),
            dmabuf.is_some(),
            dmabuf.map(|d| d.fd()),
            dmabuf.map(|d| fd_size(d.fd())),
        );
    }
    let meta = gpu_buffer.meta::<VideoMeta>();
    let (strides, offsets): (Vec<i32>, Vec<usize>) = match &meta {
        Some(meta) => (meta.stride().to_vec(), meta.offset().to_vec()),
        None => {
            println!("NO VideoMeta on the DMA-BUF: plane layout unknown");
            std::process::exit(2);
        }
    };
    println!("VideoMeta: strides={strides:?} offsets={offsets:?}");

    // CPU reference planes, tightly packed.
    let info = gstreamer_video::VideoInfo::from_caps(cpu_sample.caps().unwrap()).unwrap();
    let frame =
        gstreamer_video::VideoFrameRef::from_buffer_ref_readable(cpu_buffer, &info).unwrap();
    let packed = |plane: u32, row_bytes: usize, rows: usize| -> Vec<u8> {
        let stride = frame.plane_stride()[plane as usize] as usize;
        let data = frame.plane_data(plane).unwrap();
        (0..rows)
            .flat_map(|r| data[r * stride..r * stride + row_bytes].iter().copied())
            .collect()
    };
    let ref_y = packed(0, width as usize, height as usize);
    let ref_uv = packed(1, width as usize, height as usize / 2);

    // wgpu on Vulkan, with DMA-BUF import.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    // `WGPU_ADAPTER_NAME` picks another GPU than the decoder's, as a hybrid
    // laptop's renderer does.
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .expect("no Vulkan adapter");
    println!("GPU: {}", adapter.get_info().name);
    // What the player checks before importing: a modifier the device does not
    // list is another GPU's tiling, and imports as noise.
    let listed = unsafe {
        adapter.as_hal::<vk_hal::Api>().map(|hal| {
            let instance = hal.shared_instance().raw_instance();
            listed_modifiers(instance, hal.raw_physical_device())
        })
    }
    .unwrap_or_default();
    println!(
        "renderer lists modifier 0x{modifier:016x} for R8 and RG8: {}",
        listed.contains(&modifier)
    );
    assert!(
        adapter
            .features()
            .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF),
        "adapter lacks VULKAN_EXTERNAL_MEMORY_DMA_BUF"
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("va_dmabuf_verify"),
        required_features: wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF,
        ..Default::default()
    }))
    .expect("device");

    let planes = [
        (
            "Y",
            width,
            height,
            wgpu::TextureFormat::R8Unorm,
            1u32,
            &ref_y,
        ),
        (
            "UV",
            width / 2,
            height / 2,
            wgpu::TextureFormat::Rg8Unorm,
            2u32,
            &ref_uv,
        ),
    ];
    let mut all_pass = true;
    for (i, (label, w, h, format, bpp, reference)) in planes.into_iter().enumerate() {
        let offset = offsets[i];
        let (range, skip) = gpu_buffer
            .find_memory(offset..offset + 1)
            .expect("plane offset outside the buffer");
        let mem = gpu_buffer.peek_memory(range.start);
        let fd = mem
            .downcast_memory_ref::<DmaBufMemory>()
            .expect("plane memory is not a DMA-BUF")
            .fd();
        let plane_offset = (mem.offset() + skip) as u32;
        let got = import_and_read(
            &device,
            &queue,
            fd,
            (w, h),
            format,
            bpp,
            vk_hal::DmaBufPlaneInfo {
                drm_modifier: modifier,
                stride: strides[i] as u32,
                offset: plane_offset,
                total_size: if plane_offset > 0 { fd_size(fd) } else { 0 },
            },
            label,
        );
        let mismatches = got
            .iter()
            .zip(reference.iter())
            .filter(|(a, b)| a != b)
            .count();
        let pass = mismatches == 0 && got.len() == reference.len();
        all_pass &= pass;
        println!(
            "{label}: {} — {mismatches}/{} bytes differ (fd offset {plane_offset}, stride {})",
            if pass { "PASS" } else { "FAIL" },
            reference.len(),
            strides[i]
        );
        if !pass {
            println!("  got[0..16]={:?}", &got[..16.min(got.len())]);
            println!("  ref[0..16]={:?}", &reference[..16]);
        }
    }

    drop(frame);
    gpu_pipeline.set_state(gst::State::Null).unwrap();
    cpu_pipeline.set_state(gst::State::Null).unwrap();
    println!(
        "\n{}",
        if all_pass {
            "PASS: the VA DMA-BUF imports pixel-identical to the CPU decode."
        } else {
            "FAIL: the imported planes differ from the CPU decode."
        }
    );
    std::process::exit(if all_pass { 0 } else { 1 });
}

/// The `nth` sample (0-based) out of `appsink name=sink`, with its pipeline
/// held so the buffer is not recycled under us.
fn pull_nth(description: &str, nth: usize) -> (gst::Sample, gst::Pipeline) {
    let pipeline = gst::parse::launch(description)
        .unwrap()
        .downcast::<gst::Pipeline>()
        .unwrap();
    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<gst_app::AppSink>()
        .unwrap();
    // A VA decoder hands out a DMA-BUF only to a sink that reads its plane
    // layout from VideoMeta, and says so in the allocation query.
    sink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .propose_allocation(|_, query| {
                query.add_allocation_meta::<VideoMeta>(None);
                true
            })
            .build(),
    );
    pipeline.set_state(gst::State::Playing).unwrap();
    let mut sample = None;
    for _ in 0..=nth {
        sample = Some(
            sink.try_pull_sample(gst::ClockTime::from_seconds(10))
                .unwrap_or_else(|| panic!("no sample from: {description}")),
        );
    }
    pipeline.set_state(gst::State::Paused).unwrap();
    (sample.unwrap(), pipeline)
}

/// The modifiers both NV12 plane formats list on `physical`.
fn listed_modifiers(instance: &ash::Instance, physical: ash::vk::PhysicalDevice) -> Vec<u64> {
    use ash::vk;
    let of = |format: vk::Format| -> Vec<u64> {
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut props = vk::FormatProperties2::default().push_next(&mut list);
        unsafe { instance.get_physical_device_format_properties2(physical, format, &mut props) };
        let mut entries = vec![
            vk::DrmFormatModifierPropertiesEXT::default();
            list.drm_format_modifier_count as usize
        ];
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut entries);
        let mut props = vk::FormatProperties2::default().push_next(&mut list);
        unsafe { instance.get_physical_device_format_properties2(physical, format, &mut props) };
        entries.iter().map(|e| e.drm_format_modifier).collect()
    };
    let uv = of(vk::Format::R8G8_UNORM);
    of(vk::Format::R8_UNORM)
        .into_iter()
        .filter(|m| uv.contains(m))
        .collect()
}

fn fd_size(fd: i32) -> u64 {
    // A DMA-BUF's size is where SEEK_END lands.
    let size = unsafe { libc::lseek(fd, 0, libc::SEEK_END) };
    unsafe { libc::lseek(fd, 0, libc::SEEK_SET) };
    size.max(0) as u64
}

#[allow(clippy::too_many_arguments)]
fn import_and_read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fd: i32,
    (width, height): (u32, u32),
    format: wgpu::TextureFormat,
    bpp: u32,
    plane: vk_hal::DmaBufPlaneInfo,
    label: &str,
) -> Vec<u8> {
    let fd = unsafe { libc::dup(fd) };
    assert!(fd >= 0);
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let desc = wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };
    let texture = unsafe {
        let hal = device.as_hal::<vk_hal::Api>().expect("not Vulkan");
        let raw = hal
            .texture_from_dmabuf_fd(fd, &hal_desc, &plane)
            .unwrap_or_else(|e| panic!("{label}: import failed: {e:?}"));
        device.create_texture_from_hal::<vk_hal::Api>(raw, &desc)
    };

    let row = width * bpp;
    let aligned = row.div_ceil(ALIGN) * ALIGN;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: aligned as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned),
                rows_per_image: Some(height),
            },
        },
        size,
    );
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = slice.get_mapped_range();
    (0..height as usize)
        .flat_map(|r| data[r * aligned as usize..r * aligned as usize + row as usize].to_vec())
        .collect()
}
