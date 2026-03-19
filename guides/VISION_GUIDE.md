# Vision Guide: AI Game Emulation with Computer Vision

This guide documents patterns from reinforcement learning projects that use vision for game emulation, particularly useful for AI-PyBoy Game Boy emulation projects.

---

## Table of Contents

1. [Key Projects Reference](#key-projects-reference)
2. [Screen Analysis Patterns](#screen-analysis-patterns)
3. [Game State Detection](#game-state-detection)
4. [Timing & Frame Handling](#timing--frame-handling)
5. [Memory vs Vision Approaches](#memory-vs-vision-approaches)
6. [Architecture Patterns](#architecture-patterns)
7. [Implementation Recommendations](#implementation-recommendations)

---

## Key Projects Reference

### 1. ViZDoom (Farama Foundation)
**Repository:** https://github.com/Farama-Foundation/ViZDoom

The gold standard for vision-based RL on Doom. Key features:
- **Screen buffer** - Raw pixel input (what player sees)
- **Depth buffer** - 3D vision capability
- **Automatic object labeling** - Game objects categorized in frame
- **Game variables** - Health, ammo, etc. accessible
- **Audio buffer** - Sound input available
- **Gymnasium wrapper** - OpenAI Gym compatible
- **Fast rendering** - Up to 7000 frames/second

**Best for:** Learning vision-based RL on retro FPS games

### 2. Arnold (glample)
**Repository:** https://github.com/glample/Arnold

PyTorch implementation that won the 2017 ViZDoom competition. Key architecture:
- **DQN and DRQN** - Both feedforward and recurrent variants
- **LSTM/GRU recurrence** - For temporal dependencies
- **Dueling network option** - For value decomposition
- **Auxiliary tasks** - Predict game features (target, enemy)
- **Configurable resolution** - Default 60x108 (reduced from original)

**Typical config:**
```
--height 60 --width 108
--gray "false"
--hist_size 4  # frame history
--frame_skip 4  # skip every 4 frames
--network_type "dqn_rnn"
--recurrence "lstm"
```

### 3. Mario Kart CNN (mkdl)
**Repository:** https://https://github.com/SimiPro/mkdl

Game console emulation via Bizhawk emulator:
- **Emulator integration** - Bizhawk + Lua scripts
- **Python server architecture** - Socket communication
- **Gym-style environment** - reset(), step(), close()
- **CNN architecture** - OurCNN2 (CNN + 2 ReLU layers)
- **Agents** - PPO, A2C, A3C supported
- **Action space** - Steering directions + jump

### 4. PySC2 (DeepMind)
**Repository:** https://github.com/google-deepmind/pysc2

StarCraft II Learning Environment - complex RTS patterns:
- **Feature layers** - Not raw pixels (unit types, health, resources)
- **Minimap + main screen** - Separate observations
- **Spatial actions** - Point-and-click on screen
- **Non-spatial actions** - Build, research, select
- **Rich state information** - Full game API access

---

## Screen Analysis Patterns

### Resolution Selection

| Project | Resolution | Notes |
|---------|------------|-------|
| Arnold (Doom) | 60x108 | Reduced from native for speed |
| ViZDoom default | 320x240 | Configurable |
| PySC2 | Variable | Feature layers at various resolutions |
| Mario Kart | Native | Emulator captures full resolution |

**Recommendation for Game Boy:** Start with native 160x144, reduce to 80x72 for speed if needed.

### Grayscale vs Color

- **Grayscale** - Faster training, less data
- **Color** - More information, better for color-coded games
- **Game Boy** - Native is 4-color grayscale, perfect for vision!

### Frame History (Temporal Stacking)

Most successful implementations stack multiple frames:

```python
# Typical configuration
hist_size = 4  # Stack 4 frames
frame_skip = 4  # Skip 4 frames between actions

# This gives the agent information about:
# - Object movement direction
# - Animation states
# - Timing windows
```

### Preprocessing Pipeline

```python
def preprocess_screen(screen_buffer, target_size=(80, 72)):
    """Standard preprocessing from ViZDoom/Arnold"""
    # 1. Resize to target
    resized = cv2.resize(screen_buffer, target_size)
    
    # 2. Convert to grayscale (if needed)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # 3. Normalize pixel values
    normalized = gray / 255.0
    
    # 4. Stack with history
    return np.stack([normalized] * hist_size, axis=0)
```

---

## Game State Detection

### Direct Variable Access (Memory-Based)

| Method | Pros | Cons |
|--------|------|------|
| Memory reading | Exact values, fast | Game-specific, may not be available |
| Screen OCR | General purpose | Slow, error-prone |
| Vision classification | Flexible | Needs training data |

**Game Boy Memory Locations (example):**
```
Pokemon Red/Blue:
- Player X position: 0xD362
- Player Y position: 0xD361
- Current HP: 0xD16C (current), 0xD16D (max)
- Money: 0xD347 - 0xD349
```

### Vision-Based State Detection

**Object Detection Patterns:**

1. **Color-based detection** - For UI elements
```python
def detect_health_bar(screen):
    """Detect health bar using color threshold"""
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    # Green range for health
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return cv2.countNonZero(mask)
```

2. **Template matching** - For known UI elements
```python
def find_menu_option(screen, template):
    """Find menu option using template matching"""
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc if max_val > 0.8 else None
```

3. **CNN classification** - For complex state
```python
def classify_battle_state(screen):
    """Classify what kind of battle scene"""
    # Train CNN to classify: wild_battle, trainer_battle, menu, overworld
    return battle_classifier.predict(screen)
```

---

## Timing & Frame Handling

### Frame Skipping

Frame skipping is critical for efficient training:

```python
class FrameSkipper:
    def __init__(self, skip_frames=4):
        self.skip_frames = skip_frames
    
    def get_action_result(self, action):
        """Execute action for N frames, return final state"""
        total_reward = 0
        for _ in range(self.skip_frames):
            state, reward, done = self.env.step(action)
            total_reward += reward
            if done:
                break
        return state, total_reward, done
```

**Why skip frames:**
- Game runs too fast for RL (60fps vs 1-4 actions/sec)
- Reduces computation
- Makes temporal difference learning more stable

### Action Repeat

```python
# In many implementations, the same action is repeated
# for the entire skip period
def act(self, action):
    repeated_action = [action] * self.frame_skip
    return self.env.step(repeated_action)
```

### Timing Synchronization

```python
class GameTiming:
    def __init__(self, fps=60):
        self.target_frame_time = 1.0 / fps
        self.frame_count = 0
    
    def wait_for_frame(self):
        """Maintain stable frame rate"""
        elapsed = time.time() - self.frame_start
        sleep_time = self.target_frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
```

---

## Memory vs Vision Approaches

### Comparison

| Aspect | Memory Access | Pure Vision |
|--------|--------------|-------------|
| **Speed** | Instant | Requires inference |
| **Accuracy** | Exact | Approximate |
| **Generalization** | Game-specific | Works across games |
| **Training Data** | Not needed | Requires samples |
| **Cheat Detection** | Fails | Works legitimately |

### Hybrid Approach (Recommended)

Most successful implementations use both:

```python
class HybridStateProvider:
    def __init__(self, use_memory=True, use_vision=True):
        self.use_memory = use_memory
        self.use_vision = use_vision
        self.memory_state = None
        self.vision_state = None
    
    def get_state(self):
        """Combine memory and vision state"""
        if self.use_memory:
            self.memory_state = self.read_game_memory()
        
        if self.use_vision:
            self.vision_state = self.capture_screen()
        
        if self.use_memory and self.use_vision:
            return {
                'memory': self.memory_state,
                'vision': self.vision_state,
                'combined': self.fuse_states()
            }
        elif self.use_memory:
            return self.memory_state
        else:
            return self.vision_state
```

### When to Use What

| Scenario | Best Approach |
|----------|---------------|
| Pokemon (known memory map) | Memory + Vision hybrid |
| Unknown game | Pure vision |
| Speed-critical | Memory only |
| General AI (any game) | Vision only |

---

## Architecture Patterns

### DQN (Deep Q-Network)

```python
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_output(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def forward(self, x):
        conv_out = self.conv(x)
        return self.fc(conv_out)
```

### DRQN (Deep Recurrent Q-Network)

```python
class DRQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, n_actions)
    
    def forward(self, x, hidden=None):
        conv_out = self.conv(x)
        lstm_out, hidden = self.lstm(conv_out, hidden)
        return self.fc(lstm_out), hidden
```

### Policy Gradient (PPO/A2C)

```python
class CNNPolicy(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.actor = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        features = self.conv(x)
        return self.actor(features), self.critic(features)
```

---

## Implementation Recommendations

### For AI-PyBoy (Game Boy)

Based on research, here's the recommended approach:

#### 1. Start with Hybrid Architecture
```python
# Recommended initial setup
config = {
    'use_memory': True,      # Read game state directly
    'use_vision': True,      # Also capture screen for verification
    'vision_resolution': (80, 72),  # Half native resolution
    'frame_skip': 4,
    'frame_history': 4,
    'grayscale': True,       # Game Boy is 4-color anyway
}
```

#### 2. Action Space
```python
# Game Boy has 8 directional buttons + A/B/Start/Select
# For Pokemon: movement + A (interact) + B (cancel) is sufficient
GAMEBOY_ACTIONS = [
    'NO_OP',
    'UP', 'DOWN', 'LEFT', 'RIGHT',
    'A',           # Confirm / Attack
    'B',           # Cancel / Run
    'START',       # Menu
    'SELECT',
    # Combinations
    'UP_A', 'DOWN_A', 'LEFT_A', 'RIGHT_A',
    'UP_B', 'DOWN_B', 'LEFT_B', 'RIGHT_B',
]
```

#### 3. Quick State Detection
```python
# For Pokemon Red/Blue specifically:
def detect_game_state(memory):
    """Quick state detection from memory"""
    # Check for battle
    if memory[0xD057] != 0x00:  # Battle type
        return 'BATTLE'
    # Check for menu
    if memory[0xD725] == 0x00:
        return 'MENU'
    return 'OVERWORLD'
```

#### 4. Vision Model (for verification/expansion)
```python
# When pure vision is needed:
class GameBoyVision(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # Smaller network for Game Boy resolution
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),  # 80x72 -> 40x36
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 40x36 -> 20x18
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 20x18 -> 10x9
            nn.ReLU(),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 10 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
```

---

## Key Papers & Resources

1. **ViZDoom** - "ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning" (2016)
2. **Arnold** - "Playing FPS Games with Deep Reinforcement Learning" (2016) - https://arxiv.org/abs/1609.05521
3. **PySC2** - "StarCraft II: A New Challenge for Reinforcement Learning" (2017) - https://arxiv.org/abs/1708.04782
4. **DQN** - "Human-level control through deep reinforcement learning" (Nature 2015)

---

## Summary

| Pattern | Recommendation |
|---------|---------------|
| **Input** | Start with memory, add vision for robustness |
| **Resolution** | 80x72 (half native) or 160x144 (native) |
| **Frame stack** | 4 frames with frame_skip=4 |
| **Architecture** | DRQN (LSTM) for temporal handling |
| **State detection** | Memory primary, vision for unknown elements |
| **Speed** | Memory reading is instant, vision needs GPU |

**Key insight from research:** The best implementations use hybrid approaches - memory for speed/accuracy, vision for generalization and handling unknown states.

---

*Last Updated: 2026-03-19*
*Research Sources: ViZDoom, Arnold, mkdl, PySC2 GitHub repositories*