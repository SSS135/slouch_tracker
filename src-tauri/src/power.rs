//! App-level Windows power management (EcoQoS).
//!
//! Backgrounded (window unfocused) the app runs detection at only ~1fps, so
//! opting the process into OS execution-speed throttling is harmless there and
//! lets Task Manager mark it "efficient" (the green leaf browsers/Steam show).
//! Foregrounded, throttling is cleared so the ~30fps preview runs at full speed.
//!
//! Task Manager renders the leaf when EcoQoS (ProcessPowerThrottling) and IDLE
//! priority are set together — that documented pairing is applied as a unit.

/// Toggle Windows Energy Efficiency Mode for the current process.
///
/// `enabled = true` (background/unfocused): EcoQoS on + idle priority.
/// `enabled = false` (foreground/focused): EcoQoS off + normal priority.
///
/// Repeated calls with the same value are safe — the OS just re-applies the
/// same state. Failures are logged and never panic. No-op on non-Windows.
#[cfg(windows)]
pub fn set_efficiency_mode(enabled: bool) {
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::Threading::{
        GetCurrentProcess, ProcessPowerThrottling, SetPriorityClass, SetProcessInformation,
        IDLE_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_STATE,
    };

    // Pseudo-handle (-1) for the current process; must not be closed.
    let process: HANDLE = unsafe { GetCurrentProcess() };

    // ControlMask selects which policy the call governs; StateMask enables it
    // (nonzero) or clears it (zero). Governing EXECUTION_SPEED = EcoQoS.
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask: if enabled {
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        } else {
            0
        },
    };
    let throttle = unsafe {
        SetProcessInformation(
            process,
            ProcessPowerThrottling,
            std::ptr::from_ref(&state).cast(),
            std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        )
    };
    if let Err(error) = throttle {
        log::warn!(target: "power", "SetProcessInformation(ProcessPowerThrottling) failed: {error}");
    }

    // Idle priority is the other half of the leaf pairing; normal priority
    // restores full foreground scheduling.
    let priority = if enabled {
        IDLE_PRIORITY_CLASS
    } else {
        NORMAL_PRIORITY_CLASS
    };
    if let Err(error) = unsafe { SetPriorityClass(process, priority) } {
        log::warn!(target: "power", "SetPriorityClass failed: {error}");
    }
}

#[cfg(not(windows))]
pub fn set_efficiency_mode(_enabled: bool) {}
