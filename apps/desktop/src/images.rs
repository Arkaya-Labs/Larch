//! Images in a message: the rules a pasted, dropped or picked image meets before the core takes it
//! (`App::image_add`, which checks every image again), and the conversions the platforms'
//! clipboards need. The web UI's image-core.js, ported.

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use gpui::{Image, ImageFormat};

/// `larch::session::MAX_MESSAGE_IMAGES`.
pub const MAX_IMAGES: usize = 8;
/// `larch_model::vision::MAX_BYTES`.
pub const MAX_BYTES: usize = 32 << 20;
/// A clipboard or file this large is not read at all: a BMP of an 8K screen is about 130 MB.
const MAX_RAW_BYTES: usize = 256 << 20;
/// `larch_agent::NO_VISION`, word for word (a test reads the Rust source).
pub const NO_VISION: &str = "This model can't see images. Load a vision model (a GGUF with an mmproj file beside it) to ask about pictures.";
/// What larch decodes, said when an image is none of it.
pub const TYPES: &str = "Images can be PNG, JPEG, WebP or GIF.";

/// The format of `bytes` among those larch decodes: PNG, JPEG, WebP and GIF (its first frame).
pub fn sniff(b: &[u8]) -> Option<ImageFormat> {
    let at = |i: usize, s: &[u8]| b.len() >= i + s.len() && &b[i..i + s.len()] == s;
    if at(0, &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        Some(ImageFormat::Png)
    } else if at(0, &[0xff, 0xd8, 0xff]) {
        Some(ImageFormat::Jpeg)
    } else if at(0, b"RIFF") && at(8, b"WEBP") {
        Some(ImageFormat::Webp)
    } else if at(0, b"GIF87a") || at(0, b"GIF89a") {
        Some(ImageFormat::Gif)
    } else {
        None
    }
}

/// Why an image of `size` bytes cannot join a message that has `count` already, or None when it
/// can. `vision` is whether the loaded model reads images; `why`, why its projector did not load.
pub fn refusal(count: usize, size: usize, vision: bool, why: Option<&str>) -> Option<String> {
    if !vision {
        return Some(match why {
            Some(why) => format!(
                "This model's vision projector did not load ({why}), so it can't see images."
            ),
            None => NO_VISION.into(),
        });
    }
    if count >= MAX_IMAGES {
        return Some(format!("A message can carry at most {MAX_IMAGES} images."));
    }
    if size > MAX_BYTES {
        return Some("That image is larger than 32 MB.".into());
    }
    None
}

/// The bytes to send for an image a clipboard or a file gave: PNG, JPEG, WebP and GIF as they are;
/// a BMP (how WSLg hands Linux a Windows screenshot) or a TIFF (a macOS screenshot's clipboard
/// copy) as a PNG. Anything else is refused with [`TYPES`].
pub fn readable(bytes: Vec<u8>) -> Result<(Vec<u8>, ImageFormat), String> {
    if let Some(f) = sniff(&bytes) {
        return Ok((bytes, f));
    }
    if bytes.len() > MAX_RAW_BYTES {
        return Err("That image is larger than 32 MB.".into());
    }
    let format = match image::guess_format(&bytes) {
        Ok(f @ (image::ImageFormat::Bmp | image::ImageFormat::Tiff)) => f,
        _ => return Err(TYPES.into()),
    };
    let decoded = image::load_from_memory_with_format(&bytes, format)
        .map_err(|e| format!("That image could not be read: {e}"))?;
    let mut png = Vec::new();
    decoded
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| format!("That image could not be converted: {e}"))?;
    Ok((png, ImageFormat::Png))
}

/// What a chip or a saved message shows for an image of `format`: its bytes as they are, except a
/// GIF, which shows its first frame (all the model reads) as a PNG. gpui decodes every frame of a
/// GIF at the full canvas and keeps them all, so a few KB of one-pixel frames on a 4096 x 4096
/// canvas would hold gigabytes.
pub fn preview(format: ImageFormat, bytes: Vec<u8>) -> Result<Arc<Image>, String> {
    if !matches!(format, ImageFormat::Gif) {
        return Ok(Arc::new(Image::from_bytes(format, bytes)));
    }
    let first = image::load_from_memory_with_format(&bytes, image::ImageFormat::Gif)
        .map_err(|e| format!("That image could not be read: {e}"))?;
    let mut png = Vec::new();
    first
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| format!("That image could not be converted: {e}"))?;
    Ok(Arc::new(Image::from_bytes(ImageFormat::Png, png)))
}

/// Whether a dropped or picked file is an image by its name: one [`readable`] may take.
pub fn image_path(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        matches!(
            e.to_ascii_lowercase().as_str(),
            "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tif" | "tiff"
        )
    })
}

/// A file's bytes for [`readable`], refused unread past what any image could be.
pub fn read_file(p: &Path) -> Result<Vec<u8>, String> {
    let name = p
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let len = std::fs::metadata(p)
        .map_err(|e| format!("Cannot read {name}: {e}"))?
        .len();
    if len as usize > MAX_RAW_BYTES {
        return Err(format!("{name} is larger than 32 MB."));
    }
    std::fs::read(p).map_err(|e| format!("Cannot read {name}: {e}"))
}

/// What a paste in the composer holds: text pastes as text whatever else is beside it (a web page
/// copies a picture with its words), so only a clipboard with no text attaches its image. An
/// empty string is no text: WSLg offers one for every Windows clipboard change, an image's too.
/// `entries` are a clipboard item's (`ClipboardItem::entries`).
pub fn pasted_image(entries: &[gpui::ClipboardEntry]) -> Option<Arc<Image>> {
    if (entries.iter())
        .any(|e| matches!(e, gpui::ClipboardEntry::String(t) if !t.text().is_empty()))
    {
        return None;
    }
    entries.iter().find_map(|e| match e {
        gpui::ClipboardEntry::Image(i) => Some(Arc::new(i.clone())),
        _ => None,
    })
}

/// The OS clipboard's image when the window's own read found none, as a PNG. Windows keeps a Print
/// Screen and many apps' copies as a bitmap only, which gpui does not read. Under WSL the window
/// is an X11 client of WSLg, whose clipboard bridge to X11 carries text only, so the Windows
/// clipboard is read through PowerShell. Elsewhere gpui reads every image format a screenshot is
/// copied in, and there is nothing more to try.
#[cfg(target_os = "windows")]
pub fn os_clipboard_png() -> Option<Vec<u8>> {
    let data = arboard::Clipboard::new().ok()?.get_image().ok()?;
    let rgba = image::RgbaImage::from_raw(
        u32::try_from(data.width).ok()?,
        u32::try_from(data.height).ok()?,
        data.bytes.into_owned(),
    )?;
    let mut png = Vec::new();
    image::DynamicImage::ImageRgba8(rgba)
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .ok()?;
    Some(png)
}

#[cfg(target_os = "linux")]
pub fn os_clipboard_png() -> Option<Vec<u8>> {
    if std::env::var_os("WSL_DISTRO_NAME").is_none()
        && !Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
    {
        return None;
    }
    // Windows PowerShell is in every Windows; its absolute path works when WSL leaves the
    // Windows PATH out.
    let full = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe";
    let exe = if Path::new(full).exists() {
        full
    } else {
        "powershell.exe"
    };
    let script = "Add-Type -AssemblyName System.Windows.Forms, System.Drawing; \
        $i = [System.Windows.Forms.Clipboard]::GetImage(); \
        if ($i) { $m = New-Object System.IO.MemoryStream; \
        $i.Save($m, [System.Drawing.Imaging.ImageFormat]::Png); \
        $o = [Console]::OpenStandardOutput(); $o.Write($m.ToArray(), 0, $m.Length); $o.Flush() }";
    let out = std::process::Command::new(exe)
        .args(["-NoProfile", "-NonInteractive", "-STA", "-Command", script])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    (out.status.success() && sniff(&out.stdout) == Some(ImageFormat::Png)).then_some(out.stdout)
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
pub fn os_clipboard_png() -> Option<Vec<u8>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniff_names_the_four_types_larch_decodes() {
        let png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0];
        assert_eq!(sniff(&png), Some(ImageFormat::Png));
        assert_eq!(sniff(&[0xff, 0xd8, 0xff, 0xe0]), Some(ImageFormat::Jpeg));
        let webp = [0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50];
        assert_eq!(sniff(&webp), Some(ImageFormat::Webp));
        assert_eq!(sniff(b"GIF89a"), Some(ImageFormat::Gif));
        assert_eq!(sniff(b"<svg"), None);
        assert_eq!(sniff(&[0x89, 0x50]), None);
    }

    #[test]
    fn refusal_no_vision_too_many_too_large() {
        assert!(
            refusal(0, 10, false, None)
                .unwrap()
                .contains("can't see images")
        );
        assert!(
            refusal(MAX_IMAGES, 10, true, None)
                .unwrap()
                .contains("at most 8")
        );
        assert!(
            refusal(0, MAX_BYTES + 1, true, None)
                .unwrap()
                .contains("32 MB")
        );
        assert_eq!(refusal(7, MAX_BYTES, true, None), None);
    }

    #[test]
    fn a_projector_that_did_not_load_says_why() {
        let why = refusal(0, 10, false, Some("out of memory")).unwrap();
        assert!(why.contains("did not load (out of memory)"), "{why}");
        assert_eq!(refusal(0, 10, false, None).as_deref(), Some(NO_VISION));
    }

    #[test]
    fn the_limits_and_the_no_vision_text_are_the_cores_word_for_word() {
        let run = include_str!("../../../crates/larch-agent/src/run.rs");
        let at = run
            .find("pub const NO_VISION: &str = \"")
            .expect("NO_VISION in run.rs");
        let rest = &run[at + "pub const NO_VISION: &str = \"".len()..];
        let lit = &rest[..rest.find("\";").expect("its end")];
        // A Rust string continued with `\` at a line's end drops the break and the indent.
        let joined: String = lit
            .split("\\\n")
            .enumerate()
            .map(|(i, part)| if i == 0 { part } else { part.trim_start() })
            .collect();
        assert_eq!(joined, NO_VISION);
        let session = include_str!("../../../crates/larch/src/session.rs");
        assert!(session.contains(&format!(
            "pub const MAX_MESSAGE_IMAGES: usize = {MAX_IMAGES};"
        )));
        let vision = include_str!("../../../crates/larch-model/src/vision.rs");
        assert!(vision.contains("pub const MAX_BYTES: usize = 32 << 20;"));
    }

    #[test]
    fn a_paste_with_text_pastes_the_text_and_a_screenshot_attaches() {
        let png = Image::from_bytes(ImageFormat::Png, vec![0x89, b'P', b'N', b'G']);
        let image = gpui::ClipboardItem::new_image(&png);
        assert!(pasted_image(image.entries()).is_some());
        let both = [
            gpui::ClipboardEntry::String(gpui::ClipboardString::new("words".into())),
            gpui::ClipboardEntry::Image(png),
        ];
        assert!(pasted_image(&both).is_none());
        assert!(pasted_image(gpui::ClipboardItem::new_string("x".into()).entries()).is_none());
        assert!(pasted_image(&[]).is_none());
        // WSLg's empty string beside an image is no text.
        let wslg = [
            gpui::ClipboardEntry::String(gpui::ClipboardString::new(String::new())),
            gpui::ClipboardEntry::Image(Image::from_bytes(ImageFormat::Png, vec![1])),
        ];
        assert!(pasted_image(&wslg).is_some());
    }

    #[test]
    fn a_gif_previews_as_its_first_frame_in_png() {
        let mut gif = Vec::new();
        {
            let mut enc = image::codecs::gif::GifEncoder::new(&mut gif);
            for c in [[255, 0, 0, 255], [0, 0, 255, 255]] {
                let f = image::RgbaImage::from_pixel(8, 4, image::Rgba(c));
                enc.encode_frame(image::Frame::new(f)).unwrap();
            }
        }
        let shown = preview(ImageFormat::Gif, gif).unwrap();
        assert!(matches!(shown.format, ImageFormat::Png));
        let back = image::load_from_memory(&shown.bytes).unwrap().to_rgb8();
        assert_eq!((back.width(), back.height()), (8, 4));
        assert_eq!(back.get_pixel(0, 0).0, [255, 0, 0]);
        let png = preview(ImageFormat::Png, vec![1, 2]).unwrap();
        assert_eq!(png.bytes, vec![1, 2]);
        assert!(preview(ImageFormat::Gif, b"GIF89a".to_vec()).is_err());
    }

    #[test]
    fn a_bmp_or_tiff_screenshot_becomes_a_png_and_svg_is_refused() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            3,
            2,
            image::Rgb([200, 10, 10]),
        ));
        for f in [image::ImageFormat::Bmp, image::ImageFormat::Tiff] {
            let mut b = Vec::new();
            img.write_to(&mut Cursor::new(&mut b), f).unwrap();
            let (png, format) = readable(b).unwrap();
            assert_eq!(
                (sniff(&png), format),
                (Some(ImageFormat::Png), ImageFormat::Png)
            );
            let back = image::load_from_memory(&png).unwrap().to_rgb8();
            assert_eq!(
                (back.width(), back.height(), back.get_pixel(1, 1).0),
                (3, 2, [200, 10, 10])
            );
        }
        let png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 1, 2];
        assert_eq!(readable(png.clone()).unwrap(), (png, ImageFormat::Png));
        assert_eq!(readable(b"<svg xmlns=\"x\"/>".to_vec()).unwrap_err(), TYPES);
    }

    #[test]
    fn image_files_go_by_their_names() {
        assert!(image_path(Path::new("/a/Shot.PNG")));
        assert!(image_path(Path::new("c.jpeg")));
        assert!(image_path(Path::new("scan.tiff")));
        assert!(!image_path(Path::new("paper.pdf")));
        assert!(!image_path(Path::new("folder")));
    }
}
