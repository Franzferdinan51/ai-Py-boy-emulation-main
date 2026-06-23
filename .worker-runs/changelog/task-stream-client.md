# task-stream-client

- Reworked `GameCanvas` to consume SSE frames shaped like `{ image, timestamp, frame, fps }` without requiring a synthetic `type`.
- Removed the polling fallback and added a single reconnecting EventSource lifecycle with ref-based cleanup so retries cancel on unmount.
- Updated the Web UI default backend URL and settings placeholder to `http://localhost:5002`.
- Added a Vitest/jsdom test that failed before the stream-frame fix and now passes.
