//! A 1440p frame of small text drawn at a third of its size, as a capture
//! preview or a Slate stage draws a screen recording, on whatever GPU the
//! machine has (`WGPU_ADAPTER_NAME` picks one).

use super::*;
use iced_wgpu::primitive::Pipeline as _;

const SOURCE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/media/small-text-1440p.png");
const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens/small-text-third.png");
/// Out to here: 2560x1440 at a third.
const OUT: (u32, u32) = (853, 480);

/// The frame's grey levels, row by row.
fn source() -> (u32, u32, Vec<u8>) {
    let image = image::open(SOURCE).expect("the test frame").into_luma8();
    (image.width(), image.height(), image.into_raw())
}

/// Grey as limited-range NV12: luma from the level, chroma neutral.
fn nv12(width: u32, height: u32, grey: &[u8]) -> Vec<u8> {
    let mut frame: Vec<u8> = grey
        .iter()
        .map(|&g| (16.0 + 219.0 * f32::from(g) / 255.0).round() as u8)
        .collect();
    frame.resize((width * height + width * height / 2) as usize, 128);
    frame
}

/// What a box filter makes of each third: the mean of its 3x3 source texels.
fn box_third(width: u32, grey: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity((OUT.0 * OUT.1) as usize);
    for y in 0..OUT.1 {
        for x in 0..OUT.0 {
            let mut sum = 0.0;
            for dy in 0..3 {
                for dx in 0..3 {
                    let i = ((y * 3 + dy) * width + x * 3 + dx) as usize;
                    sum += f32::from(grey[i]);
                }
            }
            out.push(sum / 9.0);
        }
    }
    out
}

/// Draw the frame into an `OUT`-sized target and read back its grey levels.
fn draw_third(width: u32, height: u32, frame: &[u8]) -> Vec<u8> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .expect("a Vulkan adapter: a GPU, or lavapipe");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("a device");
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut pipeline = VideoPipeline::new(&device, &queue, format);
    let alive = Arc::new(AtomicBool::new(true));
    pipeline.upload(
        &device,
        &queue,
        1,
        &alive,
        (width, height),
        VideoFormat::Nv12,
        frame,
        Some(width),
    );
    let (w, h) = (width as f32 / 3.0, height as f32 / 3.0);
    let bounds = iced::Rectangle::new(iced::Point::ORIGIN, iced::Size::new(w, h));
    pipeline.prepare(
        &queue,
        1,
        &(bounds * iced::Transformation::orthographic(OUT.0, OUT.1)),
        (w, h),
        (w, h),
        0.0,
    );

    let size = wgpu::Extent3d {
        width: OUT.0,
        height: OUT.1,
        depth_or_array_layers: 1,
    };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("downscale target"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let row = (OUT.0 * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("downscale readback"),
        size: u64::from(row * OUT.1),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    }));
    let clip = iced::Rectangle {
        x: 0,
        y: 0,
        width: OUT.0,
        height: OUT.1,
    };
    pipeline.draw(&view, &mut encoder, &clip, 1);
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(OUT.1),
            },
        },
        size,
    );
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU finished");
    let data = slice.get_mapped_range();
    (0..OUT.1 as usize)
        .flat_map(|y| {
            (0..OUT.0 as usize).map({
                let data = &data;
                move |x| data[y * row as usize + x * 4]
            })
        })
        .collect()
}

#[test]
fn small_text_drawn_at_a_third_reads_as_its_box_average() {
    let (width, height, grey) = source();
    let drawn = draw_third(width, height, &nv12(width, height, &grey));
    let expected = box_third(width, &grey);

    // How far each pixel is from the mean of the texels under it. One
    // bilinear tap reads 4 of the 9 and lets the rest go: text shimmers and
    // breaks up. It was 12.8 grey levels on average before the filter.
    let error = drawn
        .iter()
        .zip(&expected)
        .map(|(&got, &want)| (f32::from(got) - want).abs())
        .sum::<f32>()
        / expected.len() as f32;
    let golden = image::GrayImage::from_raw(OUT.0, OUT.1, drawn).expect("the drawn frame");
    if std::env::var_os("ICED_VIDEO_PLAYER_BLESS").is_some() {
        golden.save(GOLDEN).expect("the golden written");
    }
    assert!(
        error < 4.0,
        "a third of the frame is {error:.2} grey levels from its box average on average"
    );

    // The golden is this filter's picture, for a person to look at; another GPU
    // rounds a little differently, so it is held to the same average, not to the bit.
    let recorded = image::open(GOLDEN).expect("the golden").into_luma8();
    let drift = recorded
        .as_raw()
        .iter()
        .zip(golden.as_raw())
        .map(|(&a, &b)| f32::from(a.abs_diff(b)))
        .sum::<f32>()
        / expected.len() as f32;
    assert!(drift < 1.0, "the drawing drifted {drift:.2} levels from the golden");
}
