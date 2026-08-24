// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

//! Rate-limited `tracing` events, for warnings a per-frame path would otherwise repeat
//! thousands of times a second.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

#[inline]
#[doc(hidden)]
pub fn check_and_record(last_ns: &AtomicU64, interval_ns: u64) -> bool {
    static START: OnceLock<Instant> = OnceLock::new();
    // Zero means never fired, so the very first event must not land on it.
    let now_ns = (START.get_or_init(Instant::now).elapsed().as_nanos() as u64).max(1);
    let last = last_ns.load(Ordering::Relaxed);
    (last == 0 || now_ns.saturating_sub(last) >= interval_ns)
        && last_ns
            .compare_exchange(last, now_ns, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
}

#[doc(hidden)]
#[macro_export]
macro_rules! log_throttled {
    ($level:expr, $interval:expr, $($arg:tt)*) => {{
        if ::tracing::enabled!($level) {
            static LAST_NS: ::std::sync::atomic::AtomicU64 =
                ::std::sync::atomic::AtomicU64::new(0);
            if $crate::log::check_and_record(&LAST_NS, $interval.as_nanos() as u64) {
                ::tracing::event!($level, $($arg)*);
            }
        }
    }};
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
    use super::check_and_record;
    use std::sync::atomic::AtomicU64;
    use std::time::Duration;

    #[test]
    fn throttles_within_the_interval_then_fires_again() {
        let last = AtomicU64::new(0);
        let interval_ns = Duration::from_millis(50).as_nanos() as u64;
        assert!(check_and_record(&last, interval_ns));
        assert!(!check_and_record(&last, interval_ns));
        std::thread::sleep(Duration::from_millis(75));
        assert!(check_and_record(&last, interval_ns));
    }
}
