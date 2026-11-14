//! Bounded PDF document preparation for the vision prompt path.
//!
//! The native renderer is deliberately injected behind [`PdfRenderer`]. A
//! production implementation must run PDFium in a supervised worker and
//! enforce the supplied limits and deadline; parsing untrusted PDFs in the
//! HTTP process is not an acceptable implementation of this interface.

use std::sync::{mpsc, Arc, Mutex, MutexGuard, TryLockError};
use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::path::{Path, PathBuf};
use std::io::{self, Read, Write};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use std::io::Cursor;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::support::errors::ApiError;
use crate::support::lru::LruCache;

pub const MAX_REQUEST_DOCUMENT_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_PDF_BYTES: usize = MAX_REQUEST_DOCUMENT_BYTES;
pub const MAX_PAGES: usize = crate::protocol::MAX_IMAGE_SPANS;
pub const MAX_PAGE_PIXELS: u32 = 1024 * 1024;
pub const MAX_TEXT_CHARACTERS: usize = 1_000_000;
pub const MAX_RENDERED_BYTES: usize = 32 * 1024 * 1024;
pub const CACHE_BYTES: usize = 64 * 1024 * 1024;
pub const CACHE_ENTRIES: usize = 16;
pub const PDF_DATA_URL_PREFIX: &str = "data:application/pdf;base64,";
pub const PNG_DATA_URL_PREFIX: &str = "data:image/png;base64,";
const RENDER_WORKER_LIMIT: Duration = Duration::from_secs(30);
const WORKER_POLL: Duration = Duration::from_millis(10);
const WORKER_REQUEST_HEADER_LIMIT: usize = 4096;
const WORKER_RESPONSE_LIMIT: usize = 64 * 1024 * 1024;
#[cfg(any(target_os = "linux", target_os = "macos"))]
const WORKER_MEMORY_LIMIT_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct WorkerRequestHeader {
    version: u8,
    pdf_bytes: usize,
    limits: RenderLimits,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum WorkerResponse {
    Ok { pages: Vec<Page> },
    Error { status: u16, message: String, code: String },
}

/// One rendered PDF page, as text and a PNG data URL for image normalization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page {
    pub text: String,
    pub image_url: String,
}

impl Page {
    pub fn size(&self) -> usize {
        self.text.chars().count().saturating_mul(4).saturating_add(self.image_url.len())
    }
}

/// Per-request limits the native worker must enforce while rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderLimits {
    pub pages: usize,
    pub page_pixels: u32,
    pub text_characters: usize,
    pub rendered_bytes: usize,
    pub request_bytes: usize,
}

/// Shared request budget across source PDFs and their rendered output.
#[derive(Debug, Clone)]
pub struct DocumentBudget {
    deadline: Option<Instant>,
    pub remaining_bytes: usize,
    pub remaining_pages: usize,
}

impl DocumentBudget {
    pub fn new(deadline: Option<Instant>) -> Self {
        Self {
            deadline,
            remaining_bytes: MAX_REQUEST_DOCUMENT_BYTES,
            remaining_pages: MAX_PAGES,
        }
    }

    pub fn from_timeout(timeout: Option<Duration>) -> Self {
        Self::new(timeout.and_then(|duration| Instant::now().checked_add(duration)))
    }

    /// The worker's remaining wall budget, if this request has a deadline.
    pub fn remaining_time(&self) -> Result<Option<Duration>, ApiError> {
        self.deadline
            .map(|deadline| {
                deadline
                    .checked_duration_since(Instant::now())
                    .filter(|duration| !duration.is_zero())
                    .ok_or_else(|| ApiError::new(504, "request timed out", "request_timeout"))
            })
            .transpose()
    }

    pub fn charge(&mut self, size: usize) -> Result<(), ApiError> {
        self.remaining_time()?;
        if size > self.remaining_bytes {
            return Err(ApiError::new(
                400,
                "PDF sources and rendered pages exceed the shared request size limit",
                "invalid_request_error",
            ));
        }
        self.remaining_bytes -= size;
        Ok(())
    }

    pub fn charge_pages(&mut self, pages: usize) -> Result<(), ApiError> {
        if pages > self.remaining_pages {
            return Err(ApiError::new(400, "documents exceed the request page limit", "invalid_request_error"));
        }
        self.remaining_pages -= pages;
        Ok(())
    }

    fn limits(&self) -> RenderLimits {
        RenderLimits {
            pages: MAX_PAGES.min(self.remaining_pages),
            page_pixels: MAX_PAGE_PIXELS,
            text_characters: MAX_TEXT_CHARACTERS,
            rendered_bytes: MAX_RENDERED_BYTES,
            request_bytes: self.remaining_bytes,
        }
    }
}

/// Native PDF renderer boundary. Implementations must bound CPU/memory,
/// page count, dimensions, text, and output bytes, and honor `timeout`.
pub trait PdfRenderer: Send + Sync + 'static {
    fn render(
        &self,
        pdf: &[u8],
        limits: RenderLimits,
        timeout: Option<Duration>,
    ) -> Result<Vec<Page>, ApiError>;
}

/// A process supervisor for the PDFium adapter. The worker receives one
/// length-framed binary request and returns one length-framed JSON response.
/// Each render uses a fresh child so a timeout always terminates the entire
/// PDFium process and discards its global state.
pub struct ProcessPdfRenderer {
    executable: PathBuf,
    pdfium_library: PathBuf,
}

impl ProcessPdfRenderer {
    pub fn new(executable: impl Into<PathBuf>, pdfium_library: impl Into<PathBuf>) -> Self {
        Self { executable: executable.into(), pdfium_library: pdfium_library.into() }
    }
}

impl PdfRenderer for ProcessPdfRenderer {
    fn render(&self, pdf: &[u8], limits: RenderLimits, timeout: Option<Duration>) -> Result<Vec<Page>, ApiError> {
        if pdf.is_empty() || pdf.len() > MAX_PDF_BYTES {
            return Err(ApiError::new(400, "PDF document exceeds the size limit", "invalid_request_error"));
        }
        let caller_timeout = timeout;
        let time_limit = timeout.unwrap_or(RENDER_WORKER_LIMIT).min(RENDER_WORKER_LIMIT);
        let deadline = Instant::now().checked_add(time_limit)
            .ok_or_else(|| ApiError::new(504, "request timed out", "request_timeout"))?;
        let mut command = Command::new(&self.executable);
        command.args(["document-worker", "--pdfium"])
            .arg(&self.pdfium_library)
            // PDFium parses untrusted bytes. A parser compromise must not
            // inherit API keys or other server credentials from the parent.
            .env_clear()
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        configure_worker_limits(&mut command, time_limit);
        let mut child = command.spawn()
            .map_err(|_| ApiError::new(503, "PDF document worker could not be started", "document_unavailable"))?;
        let stdin = child.stdin.take().expect("worker stdin is piped");
        let stdout = child.stdout.take().expect("worker stdout is piped");
        let (reader_tx, reader_rx) = mpsc::channel();
        let limits_copy = limits;
        let pdf_copy = pdf.to_vec();
        let writer = thread::spawn(move || {
            write_worker_request(stdin, &pdf_copy, limits_copy)
        });
        let reader = thread::spawn(move || {
            let result = read_worker_response(stdout);
            let _ = reader_tx.send(result);
        });

        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if Instant::now() < deadline => {
                    #[cfg(target_os = "macos")]
                    match macos_worker_memory_bytes(child.id()) {
                        Ok(memory) if memory > WORKER_MEMORY_LIMIT_BYTES => {
                            terminate_worker(&mut child);
                            return Err(ApiError::new(400, "PDF processing exceeded the memory limit", "invalid_request_error"));
                        }
                        Ok(_) => {}
                        Err(_) => {
                            // The worker may have exited between try_wait
                            // above and proc_pid_rusage. Python's supervisor
                            // accepts that race and handles its exit status.
                            if let Ok(Some(status)) = child.try_wait() {
                                break Some(status);
                            }
                            terminate_worker(&mut child);
                            let _ = writer.join();
                            let _ = reader.join();
                            return Err(ApiError::new(503, "document worker memory measurement failed", "document_unavailable"));
                        }
                    }
                    thread::sleep(WORKER_POLL)
                }
                Ok(None) => break None,
                Err(_) => {
                    terminate_worker(&mut child);
                    let _ = writer.join();
                    let _ = reader.join();
                    return Err(ApiError::new(503, "PDF document worker could not be monitored", "document_unavailable"));
                }
            }
        };
        if status.is_none() {
            terminate_worker(&mut child);
            let _ = writer.join();
            let _ = reader.join();
            return Err(if caller_timeout.is_some() {
                request_timed_out()
            } else {
                ApiError::new(400, "PDF processing exceeded the time limit", "invalid_request_error")
            });
        }
        let writer_result = writer.join().unwrap_or_else(|_| Err(io::Error::other("worker request writer panicked")));
        let _ = reader.join();
        if writer_result.is_err() {
            return Err(ApiError::new(503, "PDF document worker request could not be sent", "document_worker_failed"));
        }
        if !status.expect("worker exited").success() {
            return Err(ApiError::new(503, "PDF document worker exited unexpectedly", "document_worker_failed"));
        }
        let response = reader_rx.recv().ok()
            .and_then(Result::ok)
            .ok_or_else(|| ApiError::new(503, "PDF document worker returned an invalid response", "document_worker_failed"))?;
        match response {
            WorkerResponse::Ok { pages } => {
                validate_rendered_pages(&pages, limits)?;
                Ok(pages)
            }
            WorkerResponse::Error { status, message, code } => Err(ApiError::new(status, message, code)),
        }
    }
}

fn write_worker_request(mut stdin: impl Write, pdf: &[u8], limits: RenderLimits) -> io::Result<()> {
    let header = serde_json::to_vec(&WorkerRequestHeader { version: 1, pdf_bytes: pdf.len(), limits })
        .map_err(io::Error::other)?;
    if header.len() > WORKER_REQUEST_HEADER_LIMIT {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "worker request header is too large"));
    }
    stdin.write_all(&(header.len() as u32).to_be_bytes())?;
    stdin.write_all(&header)?;
    stdin.write_all(pdf)?;
    stdin.flush()
}

fn read_worker_response(mut stdout: impl Read) -> io::Result<WorkerResponse> {
    let mut length = [0; 4];
    stdout.read_exact(&mut length)?;
    let length = u32::from_be_bytes(length) as usize;
    if length == 0 || length > WORKER_RESPONSE_LIMIT {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "worker response exceeded the size limit"));
    }
    let mut response = vec![0; length];
    stdout.read_exact(&mut response)?;
    serde_json::from_slice(&response).map_err(io::Error::other)
}

#[cfg(unix)]
fn configure_worker_limits(command: &mut Command, timeout: Duration) {
    use std::os::unix::process::CommandExt;
    let cpu_seconds = timeout.as_secs().saturating_add(u64::from(timeout.subsec_nanos() > 0)).max(1);
    unsafe {
        command.pre_exec(move || {
            let no_core = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
            if libc::setrlimit(libc::RLIMIT_CORE, &no_core) != 0 {
                return Err(io::Error::last_os_error());
            }
            let cpu = libc::rlimit { rlim_cur: cpu_seconds, rlim_max: cpu_seconds };
            if libc::setrlimit(libc::RLIMIT_CPU, &cpu) != 0 {
                return Err(io::Error::last_os_error());
            }
            #[cfg(target_os = "linux")]
            {
                let limit = libc::rlimit { rlim_cur: WORKER_MEMORY_LIMIT_BYTES, rlim_max: WORKER_MEMORY_LIMIT_BYTES };
                if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn configure_worker_limits(_command: &mut Command, _timeout: Duration) {}

/// macOS has no reliable per-process `RLIMIT_AS`, so mirror Python's
/// `proc_pid_rusage` monitor and stop the isolated worker above its physical
/// footprint/resident-size cap.
#[cfg(target_os = "macos")]
fn macos_worker_memory_bytes(pid: u32) -> io::Result<u64> {
    use std::mem::MaybeUninit;

    #[repr(C)]
    struct RusageInfoV2 {
        uuid: [u8; 16],
        user_time: u64,
        system_time: u64,
        pkg_idle_wkups: u64,
        interrupt_wkups: u64,
        pageins: u64,
        wired_size: u64,
        resident_size: u64,
        phys_footprint: u64,
        proc_start_abstime: u64,
        proc_exit_abstime: u64,
        child_user_time: u64,
        child_system_time: u64,
        child_pkg_idle_wkups: u64,
        child_interrupt_wkups: u64,
        child_pageins: u64,
        child_elapsed_abstime: u64,
        diskio_bytesread: u64,
        diskio_byteswritten: u64,
    }

    #[link(name = "proc")]
    extern "C" {
        fn proc_pid_rusage(pid: libc::c_int, flavor: libc::c_int, buffer: *mut libc::c_void) -> libc::c_int;
    }

    let mut usage = MaybeUninit::<RusageInfoV2>::uninit();
    // RUSAGE_INFO_V2 is the stable public flavor in Apple's SDK; the layout
    // above follows sys/resource.h's rusage_info_v2 declaration.
    let status = unsafe { proc_pid_rusage(pid as libc::c_int, 2, usage.as_mut_ptr().cast()) };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    let usage = unsafe { usage.assume_init() };
    Ok(usage.resident_size.max(usage.phys_footprint))
}

#[cfg(not(unix))]
fn configure_worker_limits(_command: &mut Command) {}

fn terminate_worker(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Serve exactly one request on stdin/stdout for the private worker command.
/// The parent process enforces the wall-clock limit and terminates this child.
pub fn run_pdfium_worker(pdfium_library: impl AsRef<Path>) -> io::Result<()> {
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();
    let response = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_pdfium_worker_request(&mut stdin, pdfium_library.as_ref())))
        .unwrap_or_else(|_| WorkerResponse::Error {
            status: 500,
            message: "PDF document worker failed".into(),
            code: "document_worker_failed".into(),
        });
    let bytes = serde_json::to_vec(&response).map_err(io::Error::other)?;
    if bytes.len() > WORKER_RESPONSE_LIMIT {
        return write_worker_response(&mut stdout, &WorkerResponse::Error {
            status: 400,
            message: "rendered PDF exceeds the size limit".into(),
            code: "invalid_request_error".into(),
        });
    }
    write_worker_response(&mut stdout, &response)
}

fn write_worker_response(mut output: impl Write, response: &WorkerResponse) -> io::Result<()> {
    let bytes = serde_json::to_vec(response).map_err(io::Error::other)?;
    output.write_all(&(bytes.len() as u32).to_be_bytes())?;
    output.write_all(&bytes)?;
    output.flush()
}

fn run_pdfium_worker_request(input: &mut impl Read, pdfium_library: &Path) -> WorkerResponse {
    let result = (|| -> Result<Vec<Page>, ApiError> {
        let mut length = [0; 4];
        input.read_exact(&mut length).map_err(|_| ApiError::new(400, "invalid document worker request", "invalid_request_error"))?;
        let header_len = u32::from_be_bytes(length) as usize;
        if header_len == 0 || header_len > WORKER_REQUEST_HEADER_LIMIT {
            return Err(ApiError::new(400, "invalid document worker request", "invalid_request_error"));
        }
        let mut header_bytes = vec![0; header_len];
        input.read_exact(&mut header_bytes).map_err(|_| ApiError::new(400, "invalid document worker request", "invalid_request_error"))?;
        let header: WorkerRequestHeader = serde_json::from_slice(&header_bytes)
            .map_err(|_| ApiError::new(400, "invalid document worker request", "invalid_request_error"))?;
        if header.version != 1 || header.pdf_bytes == 0 || header.pdf_bytes > MAX_PDF_BYTES {
            return Err(ApiError::new(400, "PDF document exceeds the size limit", "invalid_request_error"));
        }
        let mut pdf = vec![0; header.pdf_bytes];
        input.read_exact(&mut pdf).map_err(|_| ApiError::new(400, "invalid document worker request", "invalid_request_error"))?;
        let renderer = PdfiumRenderer::load(pdfium_library)?;
        let bounded_limits = RenderLimits {
            pages: header.limits.pages.min(MAX_PAGES),
            page_pixels: header.limits.page_pixels.min(MAX_PAGE_PIXELS),
            text_characters: header.limits.text_characters.min(MAX_TEXT_CHARACTERS),
            rendered_bytes: header.limits.rendered_bytes.min(MAX_RENDERED_BYTES),
            request_bytes: header.limits.request_bytes.min(MAX_REQUEST_DOCUMENT_BYTES),
        };
        renderer.render(&pdf, bounded_limits, Some(RENDER_WORKER_LIMIT))
    })();
    match result {
        Ok(pages) => WorkerResponse::Ok { pages },
        Err(error) => WorkerResponse::Error { status: error.status, message: error.message, code: error.code },
    }
}

/// Dynamically loaded PDFium implementation. Load it from an explicit path
/// supplied by the app bundle. Keep this adapter inside a supervised worker
/// process: PDFium's synchronous C calls cannot be safely interrupted from
/// Rust when a hostile file hangs or allocates excessively.
pub struct PdfiumRenderer {
    api: PdfiumApi,
}

impl PdfiumRenderer {
    /// Open the bundled PDFium library without linking the main server to a
    /// machine-specific PDFium installation.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ApiError> {
        let api = PdfiumApi::load(path.as_ref())?;
        api.init();
        Ok(Self { api })
    }
}

impl PdfRenderer for PdfiumRenderer {
    fn render(&self, pdf: &[u8], limits: RenderLimits, timeout: Option<Duration>) -> Result<Vec<Page>, ApiError> {
        let request_deadline = timeout.and_then(|remaining| Instant::now().checked_add(remaining));
        let worker_deadline = Instant::now()
            .checked_add(timeout.unwrap_or(RENDER_WORKER_LIMIT).min(RENDER_WORKER_LIMIT));
        let request_is_shorter_than_worker = timeout.is_some_and(|remaining| remaining <= RENDER_WORKER_LIMIT);
        let check_time = || -> Result<(), ApiError> {
            if request_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(request_timed_out());
            }
            if worker_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(if request_is_shorter_than_worker {
                    request_timed_out()
                } else {
                    ApiError::new(400, "PDF processing exceeded the time limit", "invalid_request_error")
                });
            }
            Ok(())
        };
        check_time()?;
        if pdf.is_empty() || pdf.len() > MAX_PDF_BYTES {
            return Err(ApiError::new(400, "PDF document exceeds the size limit", "invalid_request_error"));
        }
        unsafe {
            let document = (self.api.load_mem)(pdf.as_ptr().cast(), pdf.len(), std::ptr::null());
            if document.is_null() {
                return Err(pdfium_load_error((self.api.last_error)()));
            }
            let _document = PdfDocumentGuard { api: &self.api, handle: document };
            // Match pypdfium2's init_forms(): create the form environment
            // before page handles are obtained, then draw widget appearances
            // after the page contents. JavaScript/XFA remain disabled in this
            // non-interactive renderer.
            let form_type = (self.api.form_type)(document);
            let form_fill_info = PdfFormFillInfo::default();
            let form = if form_type == 0 {
                None
            } else {
                let handle = (self.api.init_form_fill)(document, &form_fill_info);
                if handle.is_null() {
                    return Err(ApiError::new(400, "PDF form environment could not be initialized", "invalid_request_error"));
                }
                Some(PdfFormGuard { api: &self.api, handle })
            };
            let count = (self.api.page_count)(document);
            if count <= 0 || count as usize > limits.pages {
                return Err(ApiError::new(400, format!("PDF documents must contain 1–{} pages", limits.pages), "invalid_request_error"));
            }
            let mut pages = Vec::with_capacity(count as usize);
            let mut total_text = 0usize;
            let mut total_bytes = 0usize;
            for index in 0..count {
                check_time()?;
                let page = (self.api.load_page)(document, index);
                if page.is_null() {
                    return Err(ApiError::new(400, "PDF document could not be decoded", "invalid_request_error"));
                }
                if let Some(form) = &form {
                    (self.api.form_after_load_page)(page, form.handle);
                }
                let _page = PdfPageGuard { api: &self.api, handle: page, form: form.as_ref().map(|form| form.handle) };
                let width = (self.api.page_width)(page) as f64;
                let height = (self.api.page_height)(page) as f64;
                if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0
                    || width.max(height) / width.min(height) > 200.0
                {
                    return Err(ApiError::new(400, "PDF page dimensions are invalid", "invalid_request_error"));
                }
                let text = extract_page_text(&self.api, page, limits.text_characters.saturating_sub(total_text))?;
                check_time()?;
                total_text = total_text.saturating_add(text.chars().count());
                if total_text > limits.text_characters {
                    return Err(ApiError::new(400, "PDF text exceeds the size limit", "invalid_request_error"));
                }
                let image_url = render_page_png(&self.api, page, width, height, limits.page_pixels, form.as_ref().map(|form| form.handle))?;
                check_time()?;
                let rendered = Page { text: format!("PDF page {}:\n{}\n", index + 1, text), image_url };
                total_bytes = total_bytes.saturating_add(rendered.size());
                if total_bytes > limits.rendered_bytes {
                    return Err(ApiError::new(400, "rendered PDF exceeds the size limit", "invalid_request_error"));
                }
                pages.push(rendered);
            }
            Ok(pages)
        }
    }
}

type FpdfHandle = *mut c_void;
type InitFn = unsafe extern "C" fn();
type LoadMemFn = unsafe extern "C" fn(*const c_void, usize, *const c_char) -> FpdfHandle;
type LastErrorFn = unsafe extern "C" fn() -> c_uint;
type PageCountFn = unsafe extern "C" fn(FpdfHandle) -> c_int;
type LoadPageFn = unsafe extern "C" fn(FpdfHandle, c_int) -> FpdfHandle;
type PageSizeFn = unsafe extern "C" fn(FpdfHandle) -> f32;
type FormTypeFn = unsafe extern "C" fn(FpdfHandle) -> c_int;
type InitFormFillFn = unsafe extern "C" fn(FpdfHandle, *const PdfFormFillInfo) -> FpdfHandle;
type FormPageFn = unsafe extern "C" fn(FpdfHandle, FpdfHandle);
type FormDrawFn = unsafe extern "C" fn(FpdfHandle, FpdfHandle, FpdfHandle, c_int, c_int, c_int, c_int, c_int, c_int);
type CloseFn = unsafe extern "C" fn(FpdfHandle);
type TextLoadFn = unsafe extern "C" fn(FpdfHandle) -> FpdfHandle;
type TextCountFn = unsafe extern "C" fn(FpdfHandle) -> c_int;
type TextGetFn = unsafe extern "C" fn(FpdfHandle, c_int, c_int, *mut u16) -> c_int;
type BitmapCreateFn = unsafe extern "C" fn(c_int, c_int, c_int) -> FpdfHandle;
type BitmapFillFn = unsafe extern "C" fn(FpdfHandle, c_int, c_int, c_int, c_int, c_uint) -> c_int;
type BitmapBufferFn = unsafe extern "C" fn(FpdfHandle) -> *mut c_void;
type BitmapStrideFn = unsafe extern "C" fn(FpdfHandle) -> c_int;
type BitmapDestroyFn = unsafe extern "C" fn(FpdfHandle);
type RenderPageFn = unsafe extern "C" fn(FpdfHandle, FpdfHandle, c_int, c_int, c_int, c_int, c_int, c_int);

/// Full version 2 layout of PDFium's form callback interface. The trailing
/// 17 callbacks were added to this version and must be present even when null;
/// PDFium reads this memory according to `version`. This renderer does not run
/// JavaScript or handle user interaction, so callbacks are null and XFA is
/// disabled. The struct remains alive until the form environment is closed.
#[repr(C)]
struct PdfFormFillInfo {
    version: c_int,
    callbacks_and_js_platform: [*mut c_void; 16],
    xfa_disabled: c_int,
    version_2_callbacks: [*mut c_void; 17],
}

impl Default for PdfFormFillInfo {
    fn default() -> Self {
        Self {
            version: 2,
            callbacks_and_js_platform: [std::ptr::null_mut(); 16],
            xfa_disabled: 1,
            version_2_callbacks: [std::ptr::null_mut(); 17],
        }
    }
}

struct PdfiumApi {
    // Retained for the process lifetime. PDFium has global state, so unloading
    // its library while handles or another renderer exist would be unsafe.
    _library: *mut c_void,
    init: InitFn,
    load_mem: LoadMemFn,
    last_error: LastErrorFn,
    page_count: PageCountFn,
    load_page: LoadPageFn,
    page_width: PageSizeFn,
    page_height: PageSizeFn,
    form_type: FormTypeFn,
    init_form_fill: InitFormFillFn,
    exit_form_fill: CloseFn,
    form_after_load_page: FormPageFn,
    form_before_close_page: FormPageFn,
    close_page: CloseFn,
    close_document: CloseFn,
    text_load_page: TextLoadFn,
    text_count_chars: TextCountFn,
    text_get_text: TextGetFn,
    text_close_page: CloseFn,
    bitmap_create: BitmapCreateFn,
    bitmap_fill_rect: BitmapFillFn,
    bitmap_get_buffer: BitmapBufferFn,
    bitmap_get_stride: BitmapStrideFn,
    bitmap_destroy: BitmapDestroyFn,
    render_page_bitmap: RenderPageFn,
    draw_forms: FormDrawFn,
}

unsafe impl Send for PdfiumApi {}
unsafe impl Sync for PdfiumApi {}

impl PdfiumApi {
    fn load(path: &Path) -> Result<Self, ApiError> {
        #[cfg(unix)]
        unsafe {
            let path = CString::new(path.as_os_str().as_encoded_bytes())
                .map_err(|_| ApiError::new(400, "PDFium path contains a nul byte", "invalid_request_error"))?;
            let library = libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            if library.is_null() {
                return Err(ApiError::new(503, "bundled PDFium library could not be loaded", "document_unavailable"));
            }
            macro_rules! symbol {
                ($name:literal, $type:ty) => {{
                    let name = CString::new($name).expect("static symbol name");
                    let address = libc::dlsym(library, name.as_ptr());
                    if address.is_null() {
                        libc::dlclose(library);
                        return Err(ApiError::new(503, concat!("bundled PDFium is missing ", $name), "document_unavailable"));
                    }
                    std::mem::transmute::<*mut c_void, $type>(address)
                }};
            }
            Ok(Self {
                _library: library,
                init: symbol!("FPDF_InitLibrary", InitFn),
                load_mem: symbol!("FPDF_LoadMemDocument64", LoadMemFn),
                last_error: symbol!("FPDF_GetLastError", LastErrorFn),
                page_count: symbol!("FPDF_GetPageCount", PageCountFn),
                load_page: symbol!("FPDF_LoadPage", LoadPageFn),
                page_width: symbol!("FPDF_GetPageWidthF", PageSizeFn),
                page_height: symbol!("FPDF_GetPageHeightF", PageSizeFn),
                form_type: symbol!("FPDF_GetFormType", FormTypeFn),
                init_form_fill: symbol!("FPDFDOC_InitFormFillEnvironment", InitFormFillFn),
                exit_form_fill: symbol!("FPDFDOC_ExitFormFillEnvironment", CloseFn),
                form_after_load_page: symbol!("FORM_OnAfterLoadPage", FormPageFn),
                form_before_close_page: symbol!("FORM_OnBeforeClosePage", FormPageFn),
                close_page: symbol!("FPDF_ClosePage", CloseFn),
                close_document: symbol!("FPDF_CloseDocument", CloseFn),
                text_load_page: symbol!("FPDFText_LoadPage", TextLoadFn),
                text_count_chars: symbol!("FPDFText_CountChars", TextCountFn),
                text_get_text: symbol!("FPDFText_GetText", TextGetFn),
                text_close_page: symbol!("FPDFText_ClosePage", CloseFn),
                bitmap_create: symbol!("FPDFBitmap_Create", BitmapCreateFn),
                bitmap_fill_rect: symbol!("FPDFBitmap_FillRect", BitmapFillFn),
                bitmap_get_buffer: symbol!("FPDFBitmap_GetBuffer", BitmapBufferFn),
                bitmap_get_stride: symbol!("FPDFBitmap_GetStride", BitmapStrideFn),
                bitmap_destroy: symbol!("FPDFBitmap_Destroy", BitmapDestroyFn),
                render_page_bitmap: symbol!("FPDF_RenderPageBitmap", RenderPageFn),
                draw_forms: symbol!("FPDF_FFLDraw", FormDrawFn),
            })
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            Err(ApiError::new(503, "dynamic PDFium loading is not supported on this platform", "document_unavailable"))
        }
    }

    fn init(&self) {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| unsafe { (self.init)() });
    }
}

struct PdfDocumentGuard<'a> { api: &'a PdfiumApi, handle: FpdfHandle }
impl Drop for PdfDocumentGuard<'_> { fn drop(&mut self) { unsafe { (self.api.close_document)(self.handle) } } }
struct PdfPageGuard<'a> { api: &'a PdfiumApi, handle: FpdfHandle, form: Option<FpdfHandle> }
impl Drop for PdfPageGuard<'_> { fn drop(&mut self) { unsafe { if let Some(form) = self.form { (self.api.form_before_close_page)(self.handle, form) } (self.api.close_page)(self.handle) } } }
struct PdfFormGuard<'a> { api: &'a PdfiumApi, handle: FpdfHandle }
impl Drop for PdfFormGuard<'_> { fn drop(&mut self) { unsafe { (self.api.exit_form_fill)(self.handle) } } }
struct PdfTextGuard<'a> { api: &'a PdfiumApi, handle: FpdfHandle }
impl Drop for PdfTextGuard<'_> { fn drop(&mut self) { unsafe { (self.api.text_close_page)(self.handle) } } }
struct PdfBitmapGuard<'a> { api: &'a PdfiumApi, handle: FpdfHandle }
impl Drop for PdfBitmapGuard<'_> { fn drop(&mut self) { unsafe { (self.api.bitmap_destroy)(self.handle) } } }

unsafe fn extract_page_text(api: &PdfiumApi, page: FpdfHandle, remaining: usize) -> Result<String, ApiError> {
    let text_page = (api.text_load_page)(page);
    if text_page.is_null() { return Err(ApiError::new(400, "PDF text could not be extracted", "invalid_request_error")); }
    let _guard = PdfTextGuard { api, handle: text_page };
    let count = (api.text_count_chars)(text_page);
    if count < 0 || count as usize > remaining {
        return Err(ApiError::new(400, "PDF text exceeds the size limit", "invalid_request_error"));
    }
    let mut utf16 = vec![0u16; count as usize + 1];
    let extracted = (api.text_get_text)(text_page, 0, count, utf16.as_mut_ptr());
    if extracted < 0 || extracted as usize > utf16.len() {
        return Err(ApiError::new(400, "PDF text could not be extracted", "invalid_request_error"));
    }
    utf16.truncate(extracted as usize);
    while utf16.last() == Some(&0) { utf16.pop(); }
    Ok(String::from_utf16_lossy(&utf16).trim().to_owned())
}

unsafe fn render_page_png(api: &PdfiumApi, page: FpdfHandle, width: f64, height: f64, page_pixels: u32, form: Option<FpdfHandle>) -> Result<String, ApiError> {
    if page_pixels == 0 { return Err(ApiError::new(400, "PDF page pixel limit is zero", "invalid_request_error")); }
    let mut scale = 2.0f64.min((page_pixels as f64 / width / height).sqrt());
    if !scale.is_finite() || scale <= 0.0 { return Err(ApiError::new(400, "PDF page dimensions are invalid", "invalid_request_error")); }
    let (mut pixel_width, mut pixel_height) = ((width * scale).ceil() as u32, (height * scale).ceil() as u32);
    while u64::from(pixel_width) * u64::from(pixel_height) > u64::from(page_pixels) {
        scale *= 0.99;
        pixel_width = (width * scale).ceil() as u32;
        pixel_height = (height * scale).ceil() as u32;
    }
    if pixel_width == 0 || pixel_height == 0 || pixel_width > i32::MAX as u32 || pixel_height > i32::MAX as u32 {
        return Err(ApiError::new(400, "PDF page dimensions are invalid", "invalid_request_error"));
    }
    const FPDF_ANNOT: c_int = 1;
    // FPDFBitmap_BGRA = 4. FPDFBitmap_Gray (1) is one byte per pixel and
    // would make the four-byte row handling below read past the bitmap.
    let bitmap = (api.bitmap_create)(pixel_width as c_int, pixel_height as c_int, 4);
    if bitmap.is_null() { return Err(ApiError::new(400, "PDF page could not be rendered", "invalid_request_error")); }
    let _guard = PdfBitmapGuard { api, handle: bitmap };
    if (api.bitmap_fill_rect)(bitmap, 0, 0, pixel_width as c_int, pixel_height as c_int, 0xFFFF_FFFF) == 0 {
        return Err(ApiError::new(400, "PDF page could not be rendered", "invalid_request_error"));
    }
    (api.render_page_bitmap)(bitmap, page, 0, 0, pixel_width as c_int, pixel_height as c_int, 0, FPDF_ANNOT);
    if let Some(form) = form {
        (api.draw_forms)(form, bitmap, page, 0, 0, pixel_width as c_int, pixel_height as c_int, 0, FPDF_ANNOT);
    }
    let stride = (api.bitmap_get_stride)(bitmap);
    let buffer = (api.bitmap_get_buffer)(bitmap).cast::<u8>();
    let row_bytes = (pixel_width as usize).saturating_mul(4);
    if stride < 0 || (stride as usize) < row_bytes || buffer.is_null() {
        return Err(ApiError::new(400, "PDF page could not be rendered", "invalid_request_error"));
    }
    let source = std::slice::from_raw_parts(buffer, stride as usize * pixel_height as usize);
    let mut rgba = image::RgbaImage::new(pixel_width, pixel_height);
    for y in 0..pixel_height as usize {
        for x in 0..pixel_width as usize {
            let offset = y * stride as usize + x * 4;
            let pixel = image::Rgba([source[offset + 2], source[offset + 1], source[offset], source[offset + 3]]);
            rgba.put_pixel(x as u32, y as u32, pixel);
        }
    }
    let mut png = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(rgba).write_to(&mut png, image::ImageFormat::Png)
        .map_err(|_| ApiError::new(500, "PDF page could not be encoded", "document_worker_failed"))?;
    let encoded = STANDARD.encode(png.into_inner());
    Ok(format!("{PNG_DATA_URL_PREFIX}{encoded}"))
}

fn pdfium_load_error(code: c_uint) -> ApiError {
    match code {
        4 => ApiError::new(400, "PDF documents requiring an opening password are not supported", "invalid_request_error"),
        5 => ApiError::new(400, "PDF security handler is not supported", "invalid_request_error"),
        _ => ApiError::new(400, "PDF document could not be decoded", "invalid_request_error"),
    }
}

fn request_timed_out() -> ApiError {
    ApiError::new(504, "request timed out", "request_timeout")
}

/// Shared render/cache service. It charges cached and newly rendered pages
/// to each request and serializes cache misses through one bounded worker.
pub struct DocumentService {
    renderer: Arc<dyn PdfRenderer>,
    cache: Mutex<LruCache<[u8; 32], Vec<Page>>>,
    render_lock: Mutex<()>,
}

impl DocumentService {
    pub fn new(renderer: Arc<dyn PdfRenderer>) -> Self {
        Self {
            renderer,
            cache: Mutex::new(LruCache::new(CACHE_BYTES, Some(CACHE_ENTRIES))),
            render_lock: Mutex::new(()),
        }
    }

    /// Decode and render one standard-base64 inline PDF into canonical image
    /// and text parts. The caller should hold no prompt preparation slot
    /// while this method waits for the single render worker.
    pub fn pdf_content(&self, encoded: &str, budget: &mut DocumentBudget) -> Result<Vec<Value>, ApiError> {
        let pages = self.render_encoded(encoded, budget)?;
        let mut parts = Vec::with_capacity(pages.len().saturating_mul(2));
        for page in pages {
            parts.push(json!({"type": "text", "text": page.text}));
            parts.push(json!({"type": "image_url", "image_url": {"url": page.image_url}}));
        }
        Ok(parts)
    }

    /// Render a canonical OpenAI `file` content part.
    pub fn file_content(&self, file: &Value, budget: &mut DocumentBudget) -> Result<Vec<Value>, ApiError> {
        let object = file.as_object().ok_or_else(|| ApiError::new(400, "file must be an object", "invalid_request_error"))?;
        if object.get("file_id").is_some_and(|value| !value.is_null())
            || object.get("file_url").is_some_and(|value| !value.is_null())
        {
            return Err(ApiError::new(400, "file_id and file_url are not supported; use inline file_data", "invalid_request_error"));
        }
        if object.get("filename").is_some_and(|value| !value.is_null() && !value.is_string()) {
            return Err(ApiError::new(400, "filename must be a string", "invalid_request_error"));
        }
        let data = object.get("file_data").and_then(Value::as_str)
            .ok_or_else(|| ApiError::new(400, "file_data must contain a base64 PDF", "invalid_request_error"))?;
        let encoded = if data.starts_with("data:") {
            data.strip_prefix(PDF_DATA_URL_PREFIX)
                .ok_or_else(|| ApiError::new(400, "only application/pdf file data is supported", "invalid_request_error"))?
        } else { data };
        self.pdf_content(encoded, budget)
    }

    /// Expand inline PDF file parts before regular chat normalization. One
    /// document budget is shared across every file in the request.
    pub fn expand_messages(
        &self,
        messages: &Value,
        budget: &mut DocumentBudget,
        vision: bool,
    ) -> Result<Value, ApiError> {
        if !vision && messages.as_array().is_some_and(|items| items.iter().any(|message| {
            message.pointer("/content").and_then(Value::as_array).is_some_and(|parts| {
                parts.iter().any(|part| part.get("type").and_then(Value::as_str) == Some("file"))
            })
        })) {
            return Err(ApiError::new(
                400,
                "PDF input is not supported: this model is serving without vision (started with --language-only)",
                "invalid_request_error",
            ));
        }
        let Some(items) = messages.as_array() else { return Ok(messages.clone()) };
        let mut expanded = Vec::with_capacity(items.len());
        for message in items {
            let mut message = message.clone();
            let Some(content) = message.get("content").and_then(Value::as_array) else {
                expanded.push(message);
                continue;
            };
            let mut parts = Vec::with_capacity(content.len());
            for part in content {
                if part.get("type").and_then(Value::as_str) == Some("file") {
                    parts.extend(self.file_content(&part["file"], budget)?);
                } else {
                    parts.push(part.clone());
                }
            }
            if let Some(object) = message.as_object_mut() {
                object.insert("content".into(), Value::Array(parts));
            }
            expanded.push(message);
        }
        Ok(Value::Array(expanded))
    }

    /// Convert an Anthropic PDF document block into text plus a canonical
    /// file part. The prompt preparation layer renders the returned file.
    pub fn document_parts(block: &Value) -> Result<Vec<Value>, ApiError> {
        let object = block.as_object().ok_or_else(|| ApiError::new(400, "document block must be an object", "invalid_request_error"))?;
        let source = object.get("source").and_then(Value::as_object)
            .ok_or_else(|| ApiError::new(400, "documents require a base64 application/pdf source", "invalid_request_error"))?;
        if source.get("type").and_then(Value::as_str) != Some("base64")
            || source.get("media_type").and_then(Value::as_str) != Some("application/pdf")
        {
            return Err(ApiError::new(400, "documents require a base64 application/pdf source", "invalid_request_error"));
        }
        if let Some(citations) = object.get("citations").filter(|value| !value.is_null()) {
            if !citations.is_object()
                || citations.get("enabled").is_some_and(|enabled| *enabled != Value::Bool(false))
            {
                return Err(ApiError::new(400, "document citations are not supported", "invalid_request_error"));
            }
        }
        let mut parts = Vec::new();
        for field in ["title", "context"] {
            if let Some(value) = object.get(field).filter(|value| !value.is_null()) {
                let text = value.as_str().ok_or_else(|| ApiError::new(400, format!("document {field} must be a string"), "invalid_request_error"))?;
                parts.push(json!({"type": "text", "text": format!("{text}\n")}));
            }
        }
        let data = source.get("data").and_then(Value::as_str)
            .ok_or_else(|| ApiError::new(400, "PDF data must be a base64 string", "invalid_request_error"))?;
        parts.push(json!({"type": "file", "file": {"file_data": format!("{PDF_DATA_URL_PREFIX}{data}")}}));
        Ok(parts)
    }

    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.lock().expect("document cache lock");
        (cache.len(), cache.bytes)
    }

    fn render_encoded(&self, encoded: &str, budget: &mut DocumentBudget) -> Result<Vec<Page>, ApiError> {
        budget.remaining_time()?;
        if encoded.len() > 4 * ((MAX_PDF_BYTES + 2) / 3) {
            return Err(ApiError::new(400, "PDF document exceeds the size limit", "invalid_request_error"));
        }
        let pdf = STANDARD.decode(encoded)
            .map_err(|_| ApiError::new(400, "PDF data is not valid base64", "invalid_request_error"))?;
        if pdf.len() > MAX_PDF_BYTES {
            return Err(ApiError::new(400, "PDF document exceeds the size limit", "invalid_request_error"));
        }
        budget.charge(pdf.len())?;
        let key = crate::support::sha256::digest(&pdf);
        if let Some(pages) = self.cached(&key, budget)? { return Ok(pages); }

        let _worker = self.acquire_worker(budget)?;
        // Another request may have completed this PDF while we waited.
        if let Some(pages) = self.cached(&key, budget)? { return Ok(pages); }
        let limits = budget.limits();
        let timeout = budget.remaining_time()?.map(|remaining| remaining.min(RENDER_WORKER_LIMIT));
        let pages = self.renderer.render(&pdf, limits, timeout)?;
        validate_rendered_pages(&pages, limits)?;
        let rendered_size = pages.iter().map(Page::size).sum::<usize>();
        budget.charge_pages(pages.len())?;
        budget.charge(rendered_size)?;
        self.cache.lock().expect("document cache lock")
            .put(key, pages.clone(), rendered_size);
        Ok(pages)
    }

    fn cached(&self, key: &[u8; 32], budget: &mut DocumentBudget) -> Result<Option<Vec<Page>>, ApiError> {
        let pages = self.cache.lock().expect("document cache lock").get(key).cloned();
        if let Some(pages) = pages {
            budget.charge_pages(pages.len())?;
            budget.charge(pages.iter().map(Page::size).sum())?;
            Ok(Some(pages))
        } else { Ok(None) }
    }

    fn acquire_worker<'a>(&'a self, budget: &DocumentBudget) -> Result<MutexGuard<'a, ()>, ApiError> {
        loop {
            budget.remaining_time()?;
            match self.render_lock.try_lock() {
                Ok(guard) => return Ok(guard),
                Err(TryLockError::Poisoned(_)) => return Err(ApiError::new(503, "document worker is unavailable", "document_unavailable")),
                Err(TryLockError::WouldBlock) => {
                    let wait = budget.remaining_time()?.map_or(WORKER_POLL, |remaining| remaining.min(WORKER_POLL));
                    thread::sleep(wait);
                }
            }
        }
    }
}

fn validate_rendered_pages(pages: &[Page], limits: RenderLimits) -> Result<(), ApiError> {
    if pages.is_empty() || pages.len() > limits.pages {
        return Err(ApiError::new(400, format!("PDF documents must contain 1–{} pages", limits.pages), "invalid_request_error"));
    }
    let mut characters = 0usize;
    let mut bytes = 0usize;
    for page in pages {
        if !page.image_url.starts_with(PNG_DATA_URL_PREFIX) {
            return Err(ApiError::new(500, "document renderer returned a non-PNG page", "document_worker_failed"));
        }
        characters = characters.saturating_add(page.text.chars().count());
        bytes = bytes.saturating_add(page.size());
        if characters > limits.text_characters {
            return Err(ApiError::new(400, "PDF text exceeds the size limit", "invalid_request_error"));
        }
        if bytes > limits.rendered_bytes {
            return Err(ApiError::new(400, "rendered PDF exceeds the size limit", "invalid_request_error"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_form_fill_info_matches_version_two_abi_size() {
        let pointer_size = std::mem::size_of::<*mut c_void>();
        let first_pointer = (4 + pointer_size - 1) / pointer_size * pointer_size;
        let xfa_disabled = first_pointer + 16 * pointer_size;
        let second_pointer = (xfa_disabled + 4 + pointer_size - 1) / pointer_size * pointer_size;
        let expected_size = second_pointer + 17 * pointer_size;
        assert_eq!(std::mem::size_of::<PdfFormFillInfo>(), expected_size);
        assert_eq!(std::mem::offset_of!(PdfFormFillInfo, xfa_disabled), xfa_disabled);
        assert_eq!(std::mem::offset_of!(PdfFormFillInfo, version_2_callbacks), second_pointer);
    }

    /// Set `SPLASH_PDFIUM_TEST_LIBRARY` to the bundled PDFium path to run this
    /// integration check. The portable default test suite does not require a
    /// platform PDFium binary to be installed.
    #[test]
    fn pdfium_renders_acroform_widget_appearance() {
        let Ok(library) = std::env::var("SPLASH_PDFIUM_TEST_LIBRARY") else { return };
        let pdf = include_bytes!("../../dev/tests/fixtures/documents/form-widget.pdf");
        let renderer = PdfiumRenderer::load(library).unwrap();
        let pages = renderer.render(
            pdf,
            RenderLimits {
                pages: MAX_PAGES,
                page_pixels: MAX_PAGE_PIXELS,
                text_characters: MAX_TEXT_CHARACTERS,
                rendered_bytes: MAX_RENDERED_BYTES,
                request_bytes: MAX_REQUEST_DOCUMENT_BYTES,
            },
            None,
        ).unwrap();
        assert_eq!(pages.len(), 1);
        let encoded = pages[0].image_url.strip_prefix(PNG_DATA_URL_PREFIX).unwrap();
        let png = STANDARD.decode(encoded).unwrap();
        let image = image::load_from_memory(&png).unwrap().to_luma8();
        let dark_glyphs = image.enumerate_pixels()
            .filter(|(x, y, pixel)| (35..250).contains(x) && (80..150).contains(y) && pixel.0[0] < 100)
            .count();
        assert!(dark_glyphs > 20, "form value appearance was not rendered: {dark_glyphs} dark pixels");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_worker_memory_monitor_reads_current_process_usage() {
        assert!(macos_worker_memory_bytes(std::process::id()).unwrap() > 0);
    }

    #[test]
    fn worker_request_is_length_framed_and_carries_pdf_bytes() {
        let limits = RenderLimits {
            pages: 4,
            page_pixels: 512,
            text_characters: 2048,
            rendered_bytes: 4096,
            request_bytes: 8192,
        };
        let mut bytes = Vec::new();
        write_worker_request(&mut bytes, b"%PDF", limits).unwrap();
        let header_size = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let header: WorkerRequestHeader = serde_json::from_slice(&bytes[4..4 + header_size]).unwrap();
        assert_eq!(header.version, 1);
        assert_eq!(header.pdf_bytes, 4);
        assert_eq!(header.limits, limits);
        assert_eq!(&bytes[4 + header_size..], b"%PDF");
    }

    #[test]
    fn worker_response_is_length_framed_and_decoded() {
        let response = WorkerResponse::Ok {
            pages: vec![Page { text: "page".into(), image_url: format!("{PNG_DATA_URL_PREFIX}AA==") }],
        };
        let encoded = serde_json::to_vec(&response).unwrap();
        let mut frame = Vec::new();
        frame.extend_from_slice(&(encoded.len() as u32).to_be_bytes());
        frame.extend_from_slice(&encoded);
        assert!(matches!(read_worker_response(frame.as_slice()).unwrap(), WorkerResponse::Ok { pages } if pages[0].text == "page"));
    }

    #[test]
    fn worker_response_limit_matches_the_python_64_mib_budget() {
        assert_eq!(WORKER_RESPONSE_LIMIT, 64 * 1024 * 1024);
        let too_large = ((WORKER_RESPONSE_LIMIT + 1) as u32).to_be_bytes();
        match read_worker_response(&too_large[..]) {
            Err(error) => assert_eq!(error.kind(), io::ErrorKind::InvalidData),
            Ok(_) => panic!("oversized worker response was accepted"),
        }
    }
}
