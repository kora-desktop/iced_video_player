//! Which hardware decoder `playbin` picks. GStreamer ranks the first GPU's VA
//! decoders above every other GPU's, so on a laptop whose window draws on the
//! integrated GPU the frames come from the discrete one, in a tiling the
//! renderer cannot read. Once the renderer's GPU is known, its decoders are
//! ranked first.

use std::collections::HashMap;
use std::os::unix::fs::MetadataExt;
use std::sync::Mutex;

use gstreamer as gst;
use gstreamer::prelude::*;
use tracing::info;

/// A hardware decoder: its factory, the render node it decodes on and its rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Decoder {
    pub name: String,
    pub node: u64,
    pub rank: i32,
}

/// Per render node already ranked: whether it has decoders of its own.
static PREFERRED: Mutex<Option<HashMap<u64, bool>>> = Mutex::new(None);

/// Rank the decoders on render node `node` above every other GPU's, once per
/// node. `true` when `node` has hardware decoders, which then come first.
pub(crate) fn prefer_on(node: u64) -> bool {
    let mut done = PREFERRED.lock().unwrap_or_else(|e| e.into_inner());
    let done = done.get_or_insert_with(HashMap::new);
    if let Some(has) = done.get(&node) {
        return *has;
    }
    let decoders = hardware_decoders();
    for (name, rank) in ranks_for(&decoders, node) {
        if let Some(factory) = gst::ElementFactory::find(&name) {
            info!(decoder = %name, rank, "the renderer's GPU decodes first");
            factory.set_rank(gst::Rank::from(rank));
        }
    }
    let has = decoders.iter().any(|d| d.node == node);
    done.insert(node, has);
    has
}

/// The render node the hardware decoder in `pipeline` runs on, if one does.
pub(crate) fn node_in(pipeline: &gst::Pipeline) -> Option<u64> {
    pipeline
        .iterate_recurse()
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.factory()
                .is_some_and(|f| f.klass().contains("Decoder") && f.klass().contains("Hardware"))
        })
        .find_map(|e| {
            e.find_property("device-path")?;
            let path = e.property::<Option<String>>("device-path")?;
            Some(std::fs::metadata(path).ok()?.rdev())
        })
}

/// New ranks for the decoders on `node` that another GPU's decoders match or
/// outrank: one above the best of those, so the others keep their order behind.
pub(crate) fn ranks_for(decoders: &[Decoder], node: u64) -> Vec<(String, i32)> {
    let Some(elsewhere) = decoders
        .iter()
        .filter(|d| d.node != node)
        .map(|d| d.rank)
        .max()
    else {
        return Vec::new();
    };
    decoders
        .iter()
        .filter(|d| d.node == node && d.rank <= elsewhere)
        .map(|d| (d.name.clone(), elsewhere + 1))
        .collect()
}

/// Every video decoder that names the DRM device it runs on (`device-path`,
/// as the VA plugin's do).
fn hardware_decoders() -> Vec<Decoder> {
    gst::ElementFactory::factories_with_type(
        gst::ElementFactoryType::DECODER | gst::ElementFactoryType::MEDIA_VIDEO,
        gst::Rank::MARGINAL,
    )
    .into_iter()
    .filter(|f| f.klass().contains("Hardware"))
    .filter_map(|factory| {
        let element = factory.create().build().ok()?;
        element.find_property("device-path")?;
        let path = element.property::<Option<String>>("device-path")?;
        let node = std::fs::metadata(&path).ok()?.rdev();
        Some(Decoder {
            name: factory.name().to_string(),
            node,
            rank: i32::from(factory.rank()),
        })
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DISCRETE: u64 = 0xe280; // renderD128
    const INTEGRATED: u64 = 0xe281; // renderD129

    fn decoder(name: &str, node: u64, rank: i32) -> Decoder {
        Decoder {
            name: name.into(),
            node,
            rank,
        }
    }

    /// The test laptop: VA ranks the discrete GPU's decoders one above the rest.
    fn laptop() -> Vec<Decoder> {
        vec![
            decoder("vah264dec", DISCRETE, 257),
            decoder("vah265dec", DISCRETE, 257),
            decoder("varenderD129h264dec", INTEGRATED, 256),
            decoder("varenderD129h265dec", INTEGRATED, 256),
        ]
    }

    #[test]
    fn the_renderers_gpu_decodes_first() {
        let raised = ranks_for(&laptop(), INTEGRATED);
        assert_eq!(
            raised,
            vec![
                ("varenderD129h264dec".to_string(), 258),
                ("varenderD129h265dec".to_string(), 258),
            ]
        );
    }

    #[test]
    fn a_gpu_already_first_is_left_alone() {
        assert!(ranks_for(&laptop(), DISCRETE).is_empty());
    }

    #[test]
    fn one_gpu_changes_nothing() {
        let one = vec![decoder("vah264dec", INTEGRATED, 257)];
        assert!(ranks_for(&one, INTEGRATED).is_empty());
        assert!(ranks_for(&[], INTEGRATED).is_empty());
    }

    #[test]
    fn a_gpu_with_no_decoders_changes_nothing() {
        assert!(ranks_for(&laptop(), 0xe282).is_empty());
    }
}
