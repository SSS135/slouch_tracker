# Task 2025-11-10: Refactor Audio Alert to 1-Second Fixed Intervals

**STATUS:** COMPLETE

## User Request
Rework how audio alert works. Make it tick at 1 second fixed intervals instead of capture interval.

**Clarifications:**
- Play alert sound every 1 second while bad posture persists
- Keep the initial alertDelaySeconds before starting ticks
- Stop immediately when good posture detected
- Replace current single-alert behavior entirely

## Critical Discoveries

**1. Frame-based execution with timestamp-based rate limiting:**
Effect runs every frame (~30 FPS), but rate limiting checks elapsed time since last play. Simple pattern: `if (Date.now() - lastPlayTime >= 1000) play()`. No need for `setInterval` or cleanup complexity.

**2. Ref initialization matters for first alert:**
Initial `lastAlertPlayTimeRef.current = 0` ensures first alert passes the time check immediately after delay period. Alternative `null` check would require extra conditional logic.

## Solution

Added timestamp-based rate limiting to `usePostureSound.ts`:

1. **Added tracking ref** - `lastAlertPlayTimeRef` stores last play timestamp (line 34)
2. **Rate limiting check** - Only play if ≥1000ms elapsed: `timeSinceLastPlay >= 1000` (line 94)
3. **Reset on good posture** - Clear tracking when posture becomes good (line 74)
4. **Tests updated** - 4 test groups covering interval timing, rate limiting, edge cases (lines 293-448)

Alert now plays once per second during bad posture instead of spamming 30x/second.

## Related

- `tasks/2025-10-30-feature-fixed-interval-alert-delay.md` - Time-based tracking pattern foundation
- `tasks/0012-fix-consecutive-bad-frames-sound.md` - Audio restart quirks (pause + seekTo(0) + play)
