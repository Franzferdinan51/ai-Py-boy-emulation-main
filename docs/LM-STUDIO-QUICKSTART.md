# LM Studio Integration - Quick Start Guide

**⚡ Get started with local AI models in 5 minutes!**

## Prerequisites

- ✅ LM Studio installed (https://lmstudio.ai/)
- ✅ PyBoy Emulator WebUI running
- ✅ 8GB+ RAM recommended for local models

## Step 1: Download Models in LM Studio

Open LM Studio and download these recommended models:

### For Thinking (Decision Making):
- **Recommended:** `qwen3.5-35b-a3b` (balanced speed/quality)
- **Fast:** `qwen3.5-27b` (quick responses)
- **Ultra-Fast:** `glm-4.7-flash` (real-time gameplay)

### For Vision (Screen Analysis):
- **Recommended:** `qwen3-vl-8b` (best understanding)
- **Fast:** `qwen3-vl-4b` (quicker analysis)
- **Alternative:** `jan-v2-vl-high`

**How to download:**
1. Click 🔍 Search icon in LM Studio
2. Type model name (e.g., "qwen3.5-35b")
3. Click Download on the model card
4. Wait for download to complete (~2-8 GB per model)

## Step 2: Start LM Studio Server

1. Click **↔️ Local Server** icon (left sidebar)
2. Select your **thinking model** at the top
3. Click **Start Server** button
4. Note the server URL (usually `http://localhost:1234/v1`)

**✅ Server is running when you see:** "Server started" with green indicator

## Step 3: Configure PyBoy Emulator

1. Open PyBoy WebUI (http://localhost:3000 or your configured port)
2. Click **⚙️ Settings** (top-right)
3. Scroll to **"AI Provider"** section
4. Select **"LM Studio"** from AI Provider dropdown
5. Enter endpoint: `http://localhost:1234/v1`
6. Enter thinking model: `qwen3.5-35b-a3b` (or your chosen model)
7. Enter vision model: `qwen3-vl-8b` (or your chosen model)
8. Click **"Test"** button
   - ✅ Success: "LM Studio connected - X models available"
   - ❌ Error: Check LM Studio is running and URL is correct
9. Click **"Save Settings"**

## Step 4: Test the Integration

1. Load a ROM in PyBoy
2. Enable Agent Mode
3. Watch the action log for AI decisions
4. Check that actions are being taken

**Expected behavior:**
- AI analyzes screen every few seconds
- Makes decisions based on game state
- Actions appear in action log
- Game progresses automatically

## Troubleshooting

### ❌ "LM Studio not reachable"
**Fix:**
- Check LM Studio server is running (green indicator)
- Verify URL includes `/v1` suffix
- Try: `curl http://localhost:1234/v1/models`

### ❌ "No models available"
**Fix:**
- Load models in LM Studio (click model name in local server)
- Wait for model to fully load
- Refresh browser page

### ❌ "Timeout error"
**Fix:**
- Use smaller/faster models
- Increase timeout in settings (advanced)
- Check GPU memory usage in LM Studio

### ❌ Slow responses (>5 seconds)
**Fix:**
- Switch to faster model (e.g., `qwen3.5-27b`)
- Reduce GPU layers in LM Studio
- Close other applications using GPU

## Configuration Examples

### High Quality Setup
```
Endpoint: http://localhost:1234/v1
Thinking: qwen3.5-35b-a3b
Vision: qwen3-vl-8b
```
**Best for:** Complex games, strategic play

### Speed Run Setup
```
Endpoint: http://localhost:1234/v1
Thinking: qwen3.5-27b
Vision: qwen3-vl-4b
```
**Best for:** Fast-paced games, real-time response

### Balanced Setup
```
Endpoint: http://localhost:1234/v1
Thinking: unsloth/qwen3.5-35b-a3b
Vision: qwen/qwen3-vl-8b
```
**Best for:** General gameplay

## Advanced: Other OpenAI-Compatible Providers

This integration also works with:

### Ollama
```
Endpoint: http://localhost:11434/v1
Thinking: llama3.2
Vision: llava
```

### vLLM
```
Endpoint: http://localhost:8000/v1
Thinking: meta-llama/Llama-3.2-3B
Vision: llava-hf/llava-v1.6-mistral-7b-hf
```

### LocalAI
```
Endpoint: http://localhost:8080/v1
Thinking: gpt-3.5-turbo
Vision: gpt-4-vision-preview
```

## Performance Tips

1. **GPU Acceleration:** Enable in LM Studio settings
2. **Model Quantization:** Use Q4_K_M or Q5_K_M versions
3. **Context Length:** Keep at 4096-8192 for best performance
4. **Memory:** Close other apps if running low on RAM
5. **Temperature:** Lower = more focused, Higher = more creative

## Next Steps

- 📚 Read full documentation: `docs/LM-STUDIO-INTEGRATION.md`
- 🔧 Advanced configuration: Environment variables
- 🎮 Try different models for different games
- 📊 Monitor performance in LM Studio server tab

## Support

- Documentation: `/docs/LM-STUDIO-INTEGRATION.md`
- Implementation: `/LM-STUDIO-IMPLEMENTATION-SUMMARY.md`
- LM Studio Docs: https://lmstudio.ai/docs
- GitHub Issues: https://github.com/Franzferdinan51/ai-Py-boy-emulation-main/issues

---

**Last Updated:** March 19, 2026  
**Version:** 1.0.0  
**Status:** ✅ Production Ready
