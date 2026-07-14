# Robust Camera Services and Lifecycle Validation

## Summary

Fix the observed half-alive UGREEN state: the worker reports `running/ready`,
holds `/dev/video0`, has two failed peers, and never delivers a frame. Make the
monitor a hidden managed service, add OAK-D Pro preview support, and guarantee
that all locally owned child jobs terminate when the web app exits or is
killed.

## Implementation Changes

### Job and application lifecycle

- Add a generic job-process supervisor.
- Place each workload and its descendants in a tracked process group.
- Use Linux parent-death signaling plus owner PID/start-time verification so a
  web-app `SIGKILL` immediately terminates all local jobs.
- Persist supervisor and workload identities for restart cleanup.
- On graceful shutdown, send `SIGTERM`, wait up to five seconds, then escalate
  remaining processes to `SIGKILL`.
- Route every supported web launch path through the same lifecycle wrapper.

### UGREEN managed monitor

- Classify the UGREEN monitor as a managed service. Exclude it from normal Jobs
  listings, held-resource banners, and the dashboard's "Job is running" banner.
- Start it lazily, open `/dev/video0` only when media is requested, and release
  it after 15 seconds without a connected peer.
- Add a one-second heartbeat, camera and media frame timestamps, connected-peer
  counts, and concrete failure details.
- Treat a heartbeat older than five seconds, a 15-second peer-connect timeout,
  or no frames for five seconds after connection as unhealthy. Close leaked
  peers and replace the worker without blocking a request handler.
- Run a local STUN binding responder on configurable UDP port 3478 and advertise
  it to the browser, avoiding Chrome mDNS-only ICE failures on this
  multi-interface host.
- Retry an initial browser connection up to three more times after 1, 3, and 10
  seconds. After that, show the actual error until manual Retry resets the
  budget.

### RGB-D sensor previews

- Support OAK-D Pro RGB preview at 640x480/6 fps using a non-blocking DepthAI v3
  queue, latest-frame-only JPEG output, heartbeat reporting, and guaranteed
  pipeline and device closure.
- Keep snapshots as one-frame aligned 720p RGB-D captures for both RealSense and
  OAK-D Pro.
- Apply stale-worker detection to RealSense and OAK previews so dead jobs are
  not reused.
- Preserve per-device resource locks and derived artifacts under
  `working_data/`.

### Console, dependencies, and documentation

- Update the React console and rebuild the checked-in production assets while
  preserving existing worktree changes.
- Add `aioice` through `uv add` if it is imported directly by the STUN responder.
- Keep `INSTALL.md`, `scripts/install.sh`, and `docs/REWRITE_PROGRESS.md`
  synchronized with the implementation.

## Interfaces

- Add `visibility: "operator" | "service"` to persisted jobs, defaulting legacy
  records to `operator`.
- Make `GET /jobs` return operator jobs and their resources by default. Add
  `?include_services=1` for managed-service diagnostics.
- Introduce `monitor_webrtc.v2` health fields for heartbeat, camera-open state,
  capture and media counters, last-frame time, connected peers, STUN port, and
  error reason. Keep old v1 artifacts readable, but never consider them healthy
  active state.
- Keep the existing monitoring endpoint paths and offer flow compatible. Add
  health-aware ensure and retry behavior without exposing the loopback signaling
  port.

## Test Plan

### Unit and integration coverage

- Test parent-death cleanup, descendant process-group termination, persisted
  orphan recovery, service filtering, and resource retention until actual exit.
- Test STUN binding responses, mDNS-obfuscated offers, peer timeouts, heartbeat
  staleness, bounded retries, lazy camera opening, idle release, and every
  exception cleanup path.
- Test mocked DepthAI preview startup, frame advancement, failure, signal
  handling, and device and pipeline closure.
- Test stale-service replacement and stale-preview rejection through the Flask
  routes.

### Synthetic Playwright coverage

- Force Chromium mDNS ICE candidates and access the test app through
  non-loopback addresses.
- Verify video playback, advancing frames, two-peer fan-out, cleanup after
  navigation, bounded recovery, useful error text, and absence of the monitor
  from job banners.
- Cover four RGB-D cards, including OAK preview and snapshot behavior, stale
  jobs, stop-all, and restart states.

### Safety-gated hardware acceptance

- Require `--operator-authorized --allow-cameras --allow-real-robot` and record
  that no robot command or acquisition pipeline was executed.
- Assert the three RealSense IDs `033422071805`, `923322072633`, and
  `825412070181`, OAK-D Pro ID `18443010314F3B1300`, and UGREEN USB identity
  `0c45:2283`.
- Use Playwright through both `10.145.8.132` and the current Tailscale address,
  with both UGREEN peers connected.
- Run UGREEN WebRTC plus all three RealSense previews and the OAK preview
  concurrently for 30 seconds. Every frame counter must continue advancing.
- Capture and validate RGB and depth snapshots from all four RGB-D sensors.
- Repeat streaming across graceful `SIGTERM`, forced web-app `SIGKILL`, restart,
  and an individual monitor-worker crash. Require all PIDs and device handles to
  clear within five seconds after forced termination and every stream to restart
  successfully.
- Restore the normal web server afterward. Retain a timestamped JSON report and
  logs under `working_data/web_camera_acceptance/`.

### Final validation

- Run the full `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` suite.
- Run the synthetic and hardware-gated Playwright suites.
- Run frontend typecheck, lint, and production build.
- Run the installer's check-only validation and `git diff --check`.

## Assumptions

- The lab LAN and Tailscale are directly routed trusted networks. Internet or
  NAT traversal requiring TURN is out of scope.
- The authorized hardware run may interrupt and restart the currently running
  web app, but it must never contact the KUKA robot.
- ZED 2i is outside this work because it is not currently connected.
- Existing uncommitted UI and generated-asset changes belong to the user and
  must be preserved.
