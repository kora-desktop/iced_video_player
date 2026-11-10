//! Films played with no window: to their end and again, as a click on a slide's film does,
//! and frame by frame with nothing drawn.

use super::*;
use std::time::{Duration, Instant};

/// Half a second of moving pictures with no sound, written for the test.
fn film() -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("iced-video-player-eos-{}.ogv", std::process::id()));
    gst::init().expect("GStreamer");
    let pipeline = gst::parse::launch(&format!(
        "videotestsrc num-buffers=15 ! video/x-raw,width=64,height=48,framerate=30/1 \
         ! theoraenc ! oggmux ! filesink location={}",
        path.display()
    ))
    .expect("the encoder");
    pipeline.set_state(gst::State::Playing).expect("encoding");
    let bus = pipeline.bus().expect("a bus");
    let done = bus.timed_pop_filtered(
        gst::ClockTime::from_seconds(10),
        &[gst::MessageType::Eos, gst::MessageType::Error],
    );
    pipeline.set_state(gst::State::Null).expect("stopped");
    assert!(
        matches!(
            done.as_ref().map(|m| m.view()),
            Some(gst::MessageView::Eos(_))
        ),
        "the film was written: {done:?}"
    );
    path
}

/// Redraw as `VideoPlayer` does, every 16 ms, until the film ends or `within` runs out.
fn redraw_until_end(video: &mut Video, within: Duration) -> Option<Duration> {
    let start = Instant::now();
    while start.elapsed() < within {
        std::thread::sleep(Duration::from_millis(16));
        let mut inner = video.get_mut();
        if !inner.is_eos && !inner.paused() && inner.poll_end().0 {
            return Some(start.elapsed());
        }
    }
    None
}

#[test]
fn a_film_played_again_from_its_end_plays_to_its_end() {
    let path = film();
    let url = url::Url::from_file_path(&path).expect("a file URL");
    let mut video = Video::new(&url).expect("the film opens");
    assert!(
        redraw_until_end(&mut video, Duration::from_secs(10)).is_some(),
        "the first run ends"
    );
    // Every run from the end, not only the first: the end it left behind used to
    // stop every other one at once.
    for run in 1..=4 {
        std::thread::sleep(Duration::from_millis(200));
        assert!(video.eos(), "run {run} starts from the end");
        video.set_paused(false);
        let ran = redraw_until_end(&mut video, Duration::from_secs(10));
        let ran = ran.unwrap_or_else(|| panic!("run {run} never ended"));
        assert!(
            ran >= Duration::from_millis(300),
            "run {run} was stopped {ran:?} in, by the end before it"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// A player scrolled out of view draws nothing, and its frames used to be told only after a
/// draw: one `on_new_frame`, then none until it came back into view.
#[test]
fn every_frame_is_told_once_with_nothing_drawn() {
    let path = film();
    let url = url::Url::from_file_path(&path).expect("a file URL");
    let mut video = Video::new(&url).expect("the film opens");
    let mut told = 0;
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(16));
        let mut inner = video.get_mut();
        told += usize::from(inner.take_new_frame());
        assert!(!inner.take_new_frame(), "a frame is told once");
        if inner.poll_end().0 {
            break;
        }
    }
    assert!(told >= 10, "{told} of 15 frames told");
    let _ = std::fs::remove_file(&path);
}
