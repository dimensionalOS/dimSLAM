// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

//! Rate-limited `tracing` events, for warnings a per-frame path would otherwise repeat
//! thousands of times a second. Each call site throttles each of its messages on its own: a
//! message is told apart by its call site and the `key` it is given (the frames it is about,
//! say), so another frame's warning is never swallowed by the first one's.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// ns between repeats of one message; see `set_throttle_interval`.
static INTERVAL_NS: AtomicU64 = AtomicU64::new(10_000_000_000);

/// s between repeats of one throttled message, for every call site (`warn_throttle_s`).
pub fn set_throttle_interval(seconds: f64) {
    if seconds.is_finite() && seconds >= 0.0 {
        INTERVAL_NS.store((seconds * 1e9) as u64, Ordering::Relaxed);
    }
}

#[doc(hidden)]
pub fn interval_ns() -> u64 {
    INTERVAL_NS.load(Ordering::Relaxed)
}

/// One call site's messages, by key: when each last fired.
#[doc(hidden)]
pub type Fired = OnceLock<Mutex<HashMap<String, u64>>>;

#[inline]
#[doc(hidden)]
pub fn check_and_record(fired: &Fired, key: String, interval_ns: u64) -> bool {
    static START: OnceLock<Instant> = OnceLock::new();
    // Zero means never fired, so the very first event must not land on it.
    let now_ns = (START.get_or_init(Instant::now).elapsed().as_nanos() as u64).max(1);
    let Ok(mut fired) = fired.get_or_init(|| Mutex::new(HashMap::new())).lock() else {
        return true;
    };
    let last = fired.entry(key).or_insert(0);
    let due = *last == 0 || now_ns.saturating_sub(*last) >= interval_ns;
    if due {
        *last = now_ns;
    }
    due
}

#[doc(hidden)]
#[macro_export]
macro_rules! log_throttled {
    ($level:expr, key = $key:expr; $($arg:tt)*) => {{
        if ::tracing::enabled!($level) {
            static FIRED: $crate::log::Fired = ::std::sync::OnceLock::new();
            if $crate::log::check_and_record(&FIRED, format!("{:?}", $key), $crate::log::interval_ns()) {
                ::tracing::event!($level, $($arg)*);
            }
        }
    }};
    ($level:expr, $($arg:tt)*) => {
        $crate::log_throttled!($level, key = (); $($arg)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! warn_throttled {
    ($($arg:tt)*) => { $crate::log_throttled!(::tracing::Level::WARN, $($arg)*) };
}

#[doc(hidden)]
#[macro_export]
macro_rules! error_throttled {
    ($($arg:tt)*) => { $crate::log_throttled!(::tracing::Level::ERROR, $($arg)*) };
}

#[cfg(test)]
mod tests {
    use super::{check_and_record, Fired};
    use std::time::Duration;

    #[test]
    fn throttles_repeats_of_one_message_only_then_fires_again() {
        let interval_ns = Duration::from_millis(50).as_nanos() as u64;
        let fired = Fired::new();
        assert!(check_and_record(&fired, "a".into(), interval_ns));
        assert!(!check_and_record(&fired, "a".into(), interval_ns));
        assert!(check_and_record(&fired, "b".into(), interval_ns));
        std::thread::sleep(Duration::from_millis(75));
        assert!(check_and_record(&fired, "a".into(), interval_ns));
    }
}
