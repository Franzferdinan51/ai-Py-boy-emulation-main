# AI Game System

A comprehensive multi-component emulation and AI gaming system featuring Game Boy and Game Boy Advance emulators with integrated AI APIs.

![AI Game System](https://img.shields.io/badge/AI-Game%20System-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![Node.js](https://img.shields.io/badge/Node.js-18+-yellow) ![Windows](https://img.shields.io/badge/OS-Windows-blue) ![OpenClaw](https://img.shields.io/badge/OpenClaw-Ready-green)

## 🚀 Quick Start

### For OpenClaw Agents (Recommended)

OpenClaw agents can directly control the emulator via MCP tools:

```bash
# 1. Register MCP server
cd ai-pyboy-emulation/ai-game-server
mcporter add pyboy-emulator --stdio "python3 mcp_server.py"

# 2. Use in your agent
# Tools available: emulator_load_rom, emulator_press_button, emulator_get_frame, emulator_get_state, emulator_tick

# 3. Spawn a vision-powered gaming agent
sessions_spawn --task "Play Pokemon Blue - get a starter Pokemon" --model bailian/kimi-k2.5
```

See [OPENCLAW-INTEGRATION.md](./OPENCLAW-INTEGRATION.md) for full documentation.

### For Web UI (Windows)

**Windows users:** Simply run `unified_startup.bat` for an automated setup experience!

```bash
# Double-click this file to start the system
unified_startup.bat
```

## 📋 System Requirements

### Minimum Requirements
- **Operating System:** Windows 10 or later
- **Python:** 3.8 or higher
- **Node.js:** 18.0 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 500MB free space

### Recommended Requirements
- **Operating System:** Windows 11
- **Python:** 3.10 or higher
- **Node.js:** 20.0 or higher
- **RAM:** 16GB or more
- **Storage:** 1GB free space
- **Internet Connection:** For AI API access

## 🛠️ Installation Instructions

### Method 1: Automated Installation (Recommended)

1. **Download the deployment package**
2. **Extract to your desired location**
3. **Run the unified startup script:**
   ```bash
   unified_startup.bat
   ```
4. **Follow the on-screen prompts**

### Method 2: Manual Installation

1. **Install Python:**
   - Download from [python.org](https://python.org)
   - During installation, check "Add Python to PATH"
   - Verify installation: `python --version`

2. **Install Node.js:**
   - Download from [nodejs.org](https://nodejs.org)
   - During installation, check "Add to PATH"
   - Verify installation: `node --version` and `npm --version`

3. **Install Dependencies:**
   ```bash
   # Install Python dependencies
   pip install -r ai-game-server/requirements.txt

   # Install Node.js dependencies
   cd ai-game-assistant
   npm install
   cd ..
   ```

## 🎯 Quick Start Guide

### First Time Setup

1. **Run the startup script:**
   ```bash
   unified_startup.bat
   ```

2. **Choose your startup mode:**
   - **Option 1:** Basic Startup (Quick start)
   - **Option 9:** Ultimate Mode (All features enabled)

3. **Wait for services to start:**
   - Backend server: http://localhost:5000
   - Frontend application: http://localhost:5173
   - Service Monitor: http://localhost:8080 (Ultimate mode only)

### Using the Web Interface

1. **Open your browser** and navigate to: http://localhost:5173
2. **Configure AI APIs** in the settings:
   - Add your API keys for Gemini, OpenRouter, or NVIDIA NIM
   - Select your preferred AI provider
3. **Load ROM files** or use built-in games
4. **Start playing** with AI assistance!

## 🔧 AI Provider Configuration

### Overview

The AI Game System supports multiple AI providers, each with different capabilities and requirements. All providers must support **vision/image analysis** since the AI needs to analyze game screenshots to make decisions.

### Google Gemini

**Vision Capabilities**: ✅ Required
**Tool Use**: ❌ Not Required
**Model Requirements**: 
- Must support vision/image analysis
- Recommended models: `gemini-1.5-pro`, `gemini-1.5-flash`

**Setup**:
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your environment variables as `GEMINI_API_KEY`

**Features**:
- Excellent at understanding game screenshots
- Good reasoning capabilities for complex objectives
- Fast response times
- Free tier available

### OpenRouter

**Vision Capabilities**: ✅ Required (for most models)
**Tool Use**: ❌ Not Required
**Model Requirements**:
- Must support vision/image analysis
- Supports a wide variety of models including:
  - OpenAI GPT-4 Vision (`openai/gpt-4-vision-preview`)
  - Anthropic Claude (`anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`)
  - Google Gemini (`google/gemini-pro-vision`)
  - Mistral (`mistralai/mistral-large`)
  - And many more...

**Setup**:
1. Create an account at [OpenRouter](https://openrouter.ai/)
2. Get an API key from your account settings
3. Add it to your environment variables as `OPENROUTER_API_KEY`

**Features**:
- Access to a wide variety of cutting-edge models
- Competitive pricing
- No rate limits for most models
- Easy model switching

**Model Format**: 
Models must be specified in the format `vendor/model-name`. Examples:
- `openai/gpt-4-vision-preview`
- `anthropic/claude-3-opus:beta`
- `google/gemini-pro-vision`
- `mistralai/mistral-large`

### Alibaba Bailian (Recommended for OpenClaw)

**Vision Capabilities**: ✅ Required
**Tool Use**: ✅ Recommended (via OpenClaw Gateway)
**Cost**: **FREE unlimited** for most models!

Bailian is the recommended provider for OpenClaw agents with zero API costs:

| Model | Vision | Context | Best For |
|-------|--------|---------|----------|
| `bailian/kimi-k2.5` | ✅ Yes | 196K | **Primary vision** - FREE unlimited |
| `bailian/MiniMax-M2.5` | ❌ No | 196K | Planning/reasoning - FREE unlimited |
| `bailian/qwen3.5-plus` | ✅ Yes | 1M | Complex reasoning - 18K/mo quota |
| `bailian/qwen3-max` | ✅ Yes | 1M | Advanced reasoning - shared quota |
| `bailian/glm-5` | ✅ Yes | 128K | Fast coding - API credits |

**Setup for OpenClaw:**
```bash
# Already configured in OpenClaw Gateway (localhost:18789)
# Just use the model ID directly:
model = "bailian/kimi-k2.5"
```

**Features**:
- ✅ **FREE unlimited** for kimi-k2.5 and MiniMax-M2.5
- ✅ Vision-capable models for screen analysis
- ✅ Unified billing through OpenClaw
- ✅ Zero ban risk (unlike Gemini)

### NVIDIA NIM

**Vision Capabilities**: ✅ Required
**Tool Use**: ❌ Not Required
**Model Requirements**:
- Must support vision/image analysis
- Available models include:
  - `meta/llama3-8b-instruct`
  - `meta/llama3-70b-instruct`
  - `mistralai/mistral-7b-instruct-v0.2`
  - `mistralai/mixtral-8x7b-instruct-v0.1`
  - And others...

**Setup**:
1. Access NVIDIA NIM through [NVIDIA API Catalog](https://build.nvidia.com/)
2. Get an API key
3. Add it to your environment variables as `NVIDIA_API_KEY`

**Features**:
- High-performance models optimized by NVIDIA
- Fast inference times
- Access to Meta's Llama models
- Enterprise-grade reliability

### OpenAI Compatible APIs (LM Studio, Ollama, etc.)

**Vision Capabilities**: ✅ Required (for local vision models)
**Tool Use**: ❌ Not Required
**Model Requirements**:
- Must support vision/image analysis if running locally
- Can use any OpenAI-compatible model
- Popular local models:
  - `llava:13b` (vision capable)
  - `bakllava` (vision capable)
  - `gpt-4-vision-preview` (if proxied)

**Setup for LM Studio**:
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a vision-capable model (e.g., LLaVA)
3. Start the local server in LM Studio
4. The default endpoint is `http://localhost:1234/v1`
5. API key is typically not required for local use

**Setup for Ollama**:
1. Install [Ollama](https://ollama.com/)
2. Pull a vision model: `ollama pull llava`
3. Run the model: `ollama run llava`
4. The endpoint will be `http://localhost:11434/v1`

**Features**:
- Complete privacy (no data leaves your machine)
- No API costs
- Full control over model selection
- Requires sufficient local hardware (GPU recommended)

**Note**: For local providers, the API key field can typically be left blank.

## 🌐 Environment Variables

Create a `.env` file in the root directory:

```env
# AI API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # For LM Studio and other OpenAI-compatible APIs

# Server Configuration
FLASK_ENV=development
FLASK_DEBUG=true

# Port Configuration
BACKEND_PORT=5000
FRONTEND_PORT=5173
MONITOR_PORT=8080
```

## 🚨 Troubleshooting Common Issues

### Python/Node.js Not Found
**Error:** `Python is not installed or not in PATH`
**Solution:**
1. Reinstall Python with "Add to PATH" option
2. Restart your computer after installation
3. Verify with: `python --version`

### Port Already in Use
**Error:** `Port 5000 is already in use`
**Solution:**
1. Close other applications using the port
2. Use different ports in configuration
3. Run "Clean Start" option in startup script

### API Key Issues
**Error:** `Invalid API key` or `Rate limit exceeded`
**Solution:**
1. Verify your API keys are correct
2. Check your API usage and billing
3. Try a different AI provider

### Emulator ROM Issues
**Error:** `ROM file not found` or `Invalid ROM format`
**Solution:**
1. Place ROM files in the correct directory
2. Ensure ROM files are in .gb/.gba format
3. Check ROM file integrity

### Frontend Not Loading
**Error:** Frontend shows blank page or errors
**Solution:**
1. Check Node.js installation
2. Clear browser cache
3. Try a different browser
4. Run "Development Mode" for detailed error logs

## 🎮 What's Included

### Core Components
- **PyBoy:** Game Boy emulator with Python API
- **PyGBA:** Game Boy Advance emulator wrapper
- **AI Game Server:** Backend API and AI integration
- **AI Game Assistant:** React-based web interface

### Features
- **Multi-Platform Emulation:** Play Game Boy and Game Boy Advance games
- **AI Integration:** Multiple AI providers for game assistance
- **Real-time Monitoring:** Service health and performance monitoring
- **Web Interface:** Modern, responsive UI for game control
- **Comprehensive Logging:** Detailed logging and debugging tools
- **Auto-Restart:** Automatic service recovery

### Startup Modes
- **Basic Startup:** Quick start with minimal features
- **Enhanced Logging:** Comprehensive logging and monitoring
- **Monitoring Console:** Real-time service monitoring
- **Service Monitor:** Auto-restart capabilities with web dashboard
- **Development Mode:** Verbose logging and debugging
- **Ultimate Mode:** All features enabled (recommended)

## 🔍 Advanced Features

### Service Monitoring
- Real-time service health checks
- Automatic service restart on failure
- Performance metrics and logging
- Web dashboard at http://localhost:8080

### Game Wrappers
Pre-built AI interfaces for popular games:
- Pokémon series
- Mario games
- Tetris
- And more!

### Development Tools
- Hot-reload for frontend development
- Verbose logging for debugging
- Comprehensive test suites
- API documentation

## 📚 Documentation

- [Deployment Guide](README_DEPLOYMENT.md) - Detailed deployment instructions
- [Startup Guide](STARTUP_GUIDE.md) - Startup script usage
- [Monitoring Guide](SERVICE_MONITORING_GUIDE.md) - Service monitoring setup
- [API Documentation](docs/API.md) - Backend API reference

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyBoy** - Game Boy emulator core
- **mGBA** - Game Boy Advance emulator
- **React & Vite** - Frontend framework
- **Flask** - Backend API framework
- **AI Providers** - Gemini, OpenRouter, NVIDIA NIM

---

**Need Help?**
- Check the troubleshooting section above
- Review the startup script output for error messages
- Verify your API keys and network connection
- Try different startup modes for debugging

**System Status:** ✅ All components operational
**Last Updated:** September 2024
**Version:** 2.0.0