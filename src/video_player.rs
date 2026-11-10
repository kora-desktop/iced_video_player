use crate::{pipeline::VideoPrimitive, video::Video};
use iced::{
    Element,
    advanced::{self, Widget, layout, widget},
};
use iced_wgpu::primitive::Renderer as PrimitiveRenderer;
use std::{marker::PhantomData, sync::atomic::Ordering, time::Duration};
use std::{sync::Arc, time::Instant};

/// Video player widget which displays the current frame of a [`Video`](crate::Video).
#[allow(clippy::type_complexity)]
pub struct VideoPlayer<'a, Message, Theme = iced::Theme, Renderer = iced::Renderer>
where
    Renderer: PrimitiveRenderer,
{
    video: &'a Video,
    content_fit: iced::ContentFit,
    border_radius: f32,
    width: iced::Length,
    height: iced::Length,
    on_end_of_stream: Option<Message>,
    on_new_frame: Option<Message>,
    on_subtitle_text: Option<Box<dyn Fn(Option<String>) -> Message + 'a>>,
    on_error: Option<Box<dyn Fn(&glib::Error) -> Message + 'a>>,
    _phantom: PhantomData<(Theme, Renderer)>,
}

impl<'a, Message, Theme, Renderer> VideoPlayer<'a, Message, Theme, Renderer>
where
    Renderer: PrimitiveRenderer,
{
    /// Creates a new video player widget for a given video.
    pub fn new(video: &'a Video) -> Self {
        VideoPlayer {
            video,
            content_fit: iced::ContentFit::default(),
            border_radius: 0.0,
            width: iced::Length::Shrink,
            height: iced::Length::Shrink,
            on_end_of_stream: None,
            on_new_frame: None,
            on_subtitle_text: None,
            on_error: None,
            _phantom: Default::default(),
        }
    }

    /// Sets the width of the `VideoPlayer` boundaries.
    pub fn width(self, width: impl Into<iced::Length>) -> Self {
        VideoPlayer {
            width: width.into(),
            ..self
        }
    }

    /// Sets the height of the `VideoPlayer` boundaries.
    pub fn height(self, height: impl Into<iced::Length>) -> Self {
        VideoPlayer {
            height: height.into(),
            ..self
        }
    }

    /// Sets the corner radius (in logical pixels) used to round the video's
    /// corners. `0` (the default) leaves the video square.
    pub fn border_radius(self, border_radius: f32) -> Self {
        VideoPlayer {
            border_radius,
            ..self
        }
    }

    /// Sets the `ContentFit` of the `VideoPlayer`.
    pub fn content_fit(self, content_fit: iced::ContentFit) -> Self {
        VideoPlayer {
            content_fit,
            ..self
        }
    }

    /// Message to send when the video reaches the end of stream (i.e., the video ends).
    pub fn on_end_of_stream(self, on_end_of_stream: Message) -> Self {
        VideoPlayer {
            on_end_of_stream: Some(on_end_of_stream),
            ..self
        }
    }

    /// Message to send when the video receives a new frame.
    pub fn on_new_frame(self, on_new_frame: Message) -> Self {
        VideoPlayer {
            on_new_frame: Some(on_new_frame),
            ..self
        }
    }

    /// Message to send when the video receives a new frame.
    pub fn on_subtitle_text<F>(self, on_subtitle_text: F) -> Self
    where
        F: 'a + Fn(Option<String>) -> Message,
    {
        VideoPlayer {
            on_subtitle_text: Some(Box::new(on_subtitle_text)),
            ..self
        }
    }

    /// Message to send when the video playback encounters an error.
    pub fn on_error<F>(self, on_error: F) -> Self
    where
        F: 'a + Fn(&glib::Error) -> Message,
    {
        VideoPlayer {
            on_error: Some(Box::new(on_error)),
            ..self
        }
    }
}

impl<Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for VideoPlayer<'_, Message, Theme, Renderer>
where
    Message: Clone,
    Renderer: PrimitiveRenderer,
{
    fn size(&self) -> iced::Size<iced::Length> {
        iced::Size {
            width: iced::Length::Shrink,
            height: iced::Length::Shrink,
        }
    }

    fn layout(
        &mut self,
        _tree: &mut widget::Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let (video_width, video_height) = self.video.size();

        // based on `Image::layout`
        let image_size = iced::Size::new(video_width as f32, video_height as f32);
        let raw_size = limits.resolve(self.width, self.height, image_size);
        let full_size = self.content_fit.fit(image_size, raw_size);
        let final_size = iced::Size {
            width: match self.width {
                iced::Length::Shrink => f32::min(raw_size.width, full_size.width),
                _ => raw_size.width,
            },
            height: match self.height {
                iced::Length::Shrink => f32::min(raw_size.height, full_size.height),
                _ => raw_size.height,
            },
        };

        layout::Node::new(final_size)
    }

    fn draw(
        &self,
        _tree: &widget::Tree,
        renderer: &mut Renderer,
        _theme: &Theme,
        _style: &advanced::renderer::Style,
        layout: advanced::Layout<'_>,
        _cursor: advanced::mouse::Cursor,
        viewport: &iced::Rectangle,
    ) {
        let mut inner = self.video.write();

        // bounds based on `Image::draw`
        let image_size = iced::Size::new(inner.width as f32, inner.height as f32);
        let bounds = layout.bounds();
        let adjusted_fit = self.content_fit.fit(image_size, bounds.size());
        let scale = iced::Vector::new(
            adjusted_fit.width / image_size.width,
            adjusted_fit.height / image_size.height,
        );
        let final_size = image_size * scale;

        let position = match self.content_fit {
            iced::ContentFit::None => iced::Point::new(
                bounds.x + (image_size.width - adjusted_fit.width) / 2.0,
                bounds.y + (image_size.height - adjusted_fit.height) / 2.0,
            ),
            _ => iced::Point::new(
                bounds.center_x() - final_size.width / 2.0,
                bounds.center_y() - final_size.height / 2.0,
            ),
        };

        let drawing_bounds = iced::Rectangle::new(position, final_size);

        // A layer clips only to its own bounds, not to a scrollable drawn around it. Empty
        // covers edge contact and ancestors that don't cull hidden children.
        //
        // Ahead of the frame bookkeeping: this draw uploads nothing, so the flag has to keep
        // for the draw that does, and the A/V offset averages how late a frame reached the
        // screen, which a frame that is never presented has no part in.
        let Some(visible) = bounds.intersection(viewport) else {
            return;
        };

        let upload_frame = inner.upload_frame.swap(false, Ordering::SeqCst);
        if upload_frame {
            let received_at = inner
                .frame
                .lock()
                .map(|f| f.received_at)
                .unwrap_or_else(|_| Instant::now());
            inner.set_av_offset(Instant::now() - received_at);
        }

        let render = |renderer: &mut Renderer| {
            renderer.draw_primitive(
                drawing_bounds,
                VideoPrimitive::new(
                    inner.id,
                    Arc::clone(&inner.alive),
                    Arc::clone(&inner.frame),
                    (inner.width as _, inner.height as _),
                    inner.format,
                    upload_frame,
                    inner.cuda_upload.clone(),
                    self.border_radius,
                    (bounds.width, bounds.height),
                    Arc::clone(&inner.dmabuf_rejected),
                ),
            );
        };

        if adjusted_fit.width > bounds.width || adjusted_fit.height > bounds.height {
            renderer.with_layer(visible, render);
        } else {
            render(renderer);
        }
    }

    fn update(
        &mut self,
        _tree: &mut widget::Tree,
        event: &iced::Event,
        _layout: advanced::Layout<'_>,
        _cursor: advanced::mouse::Cursor,
        _renderer: &Renderer,
        shell: &mut advanced::Shell<'_, Message>,
        _viewport: &iced::Rectangle,
    ) {
        let mut inner = self.video.write();

        if let iced::Event::Window(iced::window::Event::RedrawRequested(_)) = event {
            if inner.dmabuf_rejected.load(Ordering::SeqCst) {
                inner.fall_back();
            }
            if !inner.is_eos && !inner.paused() {
                let (ended, errors) = inner.poll_end();
                for err in errors {
                    if let Some(ref on_error) = self.on_error {
                        shell.publish(on_error(&err));
                    }
                }
                if ended && let Some(on_end_of_stream) = self.on_end_of_stream.clone() {
                    shell.publish(on_end_of_stream);
                }

                // Once per frame, whether or not a draw showed it.
                let new_frame = inner.take_new_frame();
                if new_frame && let Some(on_new_frame) = self.on_new_frame.clone() {
                    shell.publish(on_new_frame);
                }

                if let Some(on_subtitle_text) = &self.on_subtitle_text
                    && inner.upload_text.swap(false, Ordering::SeqCst)
                    && let Ok(text) = inner.subtitle_text.try_lock()
                {
                    shell.publish(on_subtitle_text(text.clone()));
                }

                if new_frame {
                    // New frame ready — redraw immediately to display it.
                    shell.request_redraw();
                } else {
                    // No new frame yet — check again at roughly the next frame time.
                    // A variable frame rate has no interval to go by; poll at 60 Hz.
                    let frame_interval_ms = if inner.framerate > 0.0 {
                        1000.0 / inner.framerate
                    } else {
                        1000.0 / 60.0
                    };
                    shell.request_redraw_at(iced::window::RedrawRequest::At(next_tick(
                        Duration::from_millis(frame_interval_ms as u64),
                    )));
                }
            } else {
                // Paused or EOS — only wake when a new frame appears (e.g. seek).
                if inner.upload_frame.load(Ordering::SeqCst) {
                    shell.request_redraw();
                } else {
                    shell.request_redraw_at(iced::window::RedrawRequest::At(next_tick(
                        Duration::from_millis(250),
                    )));
                }
            }
        }
    }
}

impl<'a, Message, Theme, Renderer> From<VideoPlayer<'a, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Message: 'a + Clone,
    Theme: 'a,
    Renderer: 'a + PrimitiveRenderer,
{
    fn from(video_player: VideoPlayer<'a, Message, Theme, Renderer>) -> Self {
        Self::new(video_player)
    }
}

/// The next tick after now of a clock that started with the process, `every`
/// apart.
///
/// `iced_winit` keeps the control flow's deadline when a later one is asked
/// for, even once that deadline has passed, and polls without sleeping until
/// the later one arrives. A deadline of `now + every`, asked for again on every
/// redraw, moves later each time — and in a window that redraws for reasons of
/// its own between frames the event loop spun a core doing nothing. A tick on
/// a fixed grid is asked for again unchanged.
fn next_tick(every: Duration) -> Instant {
    static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let epoch = *EPOCH.get_or_init(Instant::now);
    let every = every.as_nanos().max(1);
    let ticks = Instant::now().duration_since(epoch).as_nanos() / every + 1;
    epoch + Duration::from_nanos((ticks * every) as u64)
}
