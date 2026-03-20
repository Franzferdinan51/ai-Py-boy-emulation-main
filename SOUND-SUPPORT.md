# Sound Support Documentation

## Overview

The PyBoy emulator now supports configurable sound output with full control over:
- Sound emulation (on/off)
- Volume control (0-100%)
- Audio output mode (speaker vs silent/headless)

## Backend API Endpoints

### GET `/api/sound/status`
Get current sound configuration and status.

**Response:**
```json
{
  "emulation_enabled": true,
  "volume": 50,
  "output_enabled": false,
  "sdl_audiodriver": "dummy",
  "sample_rate": 48000,
  "buffer_length": 1602,
  "message": "Sound output: disabled (silent mode for headless)"
}
```

### POST `/api/sound/enable`
Enable or disable sound emulation.

**Request:**
```json
{
  "enabled": true
}
```

**Note:** Changes take effect on next ROM load or reset.

### POST `/api/sound/volume`
Set sound volume (0-100).

**Request:**
```json
{
  "volume": 75
}
```

### POST `/api/sound/output`
Enable or disable actual audio output to speakers.

**Request:**
```json
{
  "enabled": true
}
```

**Note:** Changes take effect on next ROM load or reset.

### GET `/api/sound/buffer`
Get current sound buffer as base64-encoded raw audio data.

**Response:**
```json
{
  "samples": 801,
  "channels": 2,
  "sample_rate": 48000,
  "data": "base64-encoded-signed-8-bit-stereo-pcm...",
  "format": "int8_stereo"
}
```

## Sound Settings

### Sound Emulation vs Sound Output

There are two independent sound settings:

1. **Sound Emulation** (`sound_emulated`):
   - Controls whether PyBoy emulates the Game Boy sound hardware
   - When disabled, no sound buffer is generated (better performance)
   - When enabled, sound buffer is available even if output is silent

2. **Sound Output** (`SDL_AUDIODRIVER`):
   - Controls whether audio is sent to speakers
   - `'dummy'` = silent mode (no audio device needed)
   - unset = system default (actual audio output)

### Default Configuration

For headless/server mode, the defaults are:
- **Sound Emulation**: `true` (enabled)
- **Sound Output**: `false` (silent mode)
- **Volume**: `50%`
- **SDL_AUDIODRIVER**: `'dummy'`

This allows:
- AI agents to access sound buffer for audio-based decisions
- No audio device required (works on headless servers)
- No interference with other audio applications

## Platform-Specific Caveats

### macOS

1. **Headless Mode**:
   - SDL_AUDIODRIVER='dummy' is required for headless operation
   - Without it, SDL2 may crash when creating audio on background threads
   - The emulator sets this automatically

2. **Speaker Output**:
   - Requires SDL_AUDIODRIVER to be unset or set to 'coreaudio'
   - May require audio permissions in System Settings
   - Works best with a display session (not pure SSH)

3. **Known Issues**:
   - Audio may not work when running via SSH without a GUI session
   - SDL2 menu creation on background threads can cause crashes (mitigated by SDL_VIDEODRIVER='dummy')

### Linux (Headless/Server)

1. **Recommended Settings**:
   - Keep sound output disabled (`SDL_AUDIODRIVER='dummy'`)
   - Sound emulation can be enabled for buffer access

2. **PulseAudio/JACK**:
   - If speaker output is needed, ensure audio server is running
   - May need to set SDL_AUDIODRIVER='pulseaudio' or 'alsa'

3. **Docker/Containers**:
   - Sound output typically disabled in containers
   - Use dummy driver for emulation-only mode

### Windows

1. **Speaker Output**:
   - Works out of the box with default settings
   - SDL_AUDIODRIVER can be unset for DirectSound/WASAPI

2. **Remote Desktop**:
   - Audio may be redirected to RDP client
   - Works with both enabled and disabled output

## Proxy/Mobile Access

When accessing the emulator via the mobile proxy:

1. **Sound Buffer API**:
   - `/api/sound/buffer` returns raw audio data
   - Mobile app could decode and play locally
   - Currently returns base64-encoded signed 8-bit stereo PCM

2. **Latency Considerations**:
   - ~800 samples per frame at 48kHz = ~16.7ms of audio
   - Network latency may cause audio sync issues
   - Consider buffering for smooth playback

3. **Bandwidth**:
   - Raw audio: ~1600 bytes/frame * 60fps = ~96KB/s
   - Base64 encoded: ~128KB/s
   - Acceptable for local network, may be slow over WAN

## Frontend Integration

### Sound Controls

The WebUI includes sound controls in the System panel:
- Toggle for sound emulation
- Volume slider (0-100%)
- Toggle for speaker output

### Sound API Usage

```typescript
import apiService from './services/apiService';

// Get sound status
const status = await apiService.getSoundStatus();

// Enable sound emulation
await apiService.setSoundEnabled(true);

// Set volume to 75%
await apiService.setSoundVolume(75);

// Enable speaker output
await apiService.setSoundOutput(true);

// Get sound buffer for custom audio processing
const buffer = await apiService.getSoundBuffer();
```

## Troubleshooting

### No Sound on macOS

1. Check if sound output is enabled:
   ```bash
   curl http://localhost:5002/api/sound/status
   ```

2. Enable speaker output:
   ```bash
   curl -X POST http://localhost:5002/api/sound/output \
     -H "Content-Type: application/json" \
     -d '{"enabled": true}'
   ```

3. Restart the emulator (ROM reload required):
   - Reload the ROM or reset the emulator

### Audio Crackling/Stuttering

1. This may occur with actual audio output in headless mode
2. Solution: Use silent mode and fetch sound buffer separately
3. Or ensure proper audio driver is configured

### "Sound is not enabled" Error

1. Sound emulation was disabled at initialization
2. Solution: Enable sound emulation and reload ROM
3. Cannot be changed without restarting emulator

## Performance Impact

- **Sound Emulation ON**: ~1-2% CPU overhead
- **Sound Output ON**: Additional ~0.5% CPU (depends on audio backend)
- **Sound Emulation OFF**: Best performance (no audio processing)

For AI training with no audio requirements, disable sound emulation for maximum performance.

## Future Improvements

1. **WebSocket Audio Streaming**: Real-time audio streaming via WebSocket
2. **Audio Encoding**: Server-side MP3/AAC encoding for efficient transfer
3. **Audio Recording**: Save audio to file with video recording
4. **Volume Control at Runtime**: PyBoy doesn't support runtime volume changes, may need workaround