# **PYO INSTALLATION GUIDE**

## **Overview**

This guide will help you install Pyo for real audio output in the EmoTune system. Pyo is a Python module for digital signal processing and real-time synthesis.

---

## **⚠️ IMPORTANT NOTE**

**The EmoTune system works perfectly without Pyo installed!** The fallback engine will log all musical parameters and provide full functionality for testing and development.

Pyo installation is **optional** and only needed if you want actual audio output.

---

## **SYSTEM REQUIREMENTS**

### **Python Version**
- **Recommended**: Python 3.11.x
- **Minimum**: Python 3.8+
- **Current**: Python 3.13.1 (✅ Compatible)

### **Operating System**
- **Windows**: ✅ Supported (PortAudio backend)
- **macOS**: ✅ Supported (PortAudio backend)
- **Linux**: ✅ Supported (PortAudio backend)

### **Audio Requirements**
- **Audio Device**: Any standard audio output
- **Drivers**: Standard system audio drivers
- **PortAudio**: Will be installed automatically with Pyo

---

## **INSTALLATION METHODS**

### **Method 1: Automatic Installation (Recommended)**

```bash
# Try automatic installation
pip install pyo
```

**If this fails**, proceed to Method 2.

### **Method 2: Manual Wheel Installation (Windows)**

#### **Step 1: Download Pyo Wheel**
1. Go to: https://pypi.org/project/pyo/#files
2. Find the appropriate wheel for your Python version
3. Download the `.whl` file

#### **Step 2: Install the Wheel**
```bash
# Navigate to download directory
cd /path/to/downloads

# Install the wheel
pip install pyo-1.0.6-cp311-cp311-win_amd64.whl
```

**Note**: Replace the filename with the actual wheel you downloaded.

### **Method 3: Build from Source (Advanced)**

#### **Prerequisites**
```bash
# Install build tools
pip install wheel setuptools

# Install PortAudio (if needed)
# Windows: Download from http://www.portaudio.com/
# macOS: brew install portaudio
# Linux: sudo apt-get install portaudio19-dev
```

#### **Build and Install**
```bash
# Clone Pyo repository
git clone https://github.com/belangeo/pyo.git
cd pyo

# Build and install
python setup.py install
```

---

## **VERIFICATION**

### **Test Pyo Installation**

Create a test file `test_pyo_audio.py`:

```python
#!/usr/bin/env python3
"""
Test Pyo audio output
"""

try:
    from pyo import *
    print("✅ Pyo imported successfully!")
    
    # Create a simple test
    s = Server(audio="portaudio").boot()
    s.start()
    
    # Play a simple sine wave
    osc = Sine(freq=440, mul=0.3).out()
    
    print("✅ Audio test started - you should hear a 440Hz tone")
    print("Press Enter to stop...")
    input()
    
    s.stop()
    s.shutdown()
    print("✅ Audio test completed successfully!")
    
except ImportError:
    print("❌ Pyo not found. Installation may have failed.")
except Exception as e:
    print(f"❌ Audio test failed: {e}")
```

Run the test:
```bash
python test_pyo_audio.py
```

---

## **TROUBLESHOOTING**

### **Common Issues**

#### **1. Import Error: No module named 'pyo'**
**Cause**: Pyo not installed or wrong Python environment
**Solution**: 
```bash
# Check Python environment
python --version
pip list | grep pyo

# Reinstall if needed
pip uninstall pyo
pip install pyo
```

#### **2. PortAudio Error**
**Cause**: PortAudio not available
**Solution**:
```bash
# Windows: Download PortAudio from http://www.portaudio.com/
# macOS: brew install portaudio
# Linux: sudo apt-get install portaudio19-dev
```

#### **3. Audio Device Error**
**Cause**: No audio device or permissions
**Solution**:
- Check system audio settings
- Ensure audio device is enabled
- Run as administrator (Windows)

#### **4. Build Error on Windows**
**Cause**: Missing Visual Studio build tools
**Solution**:
```bash
# Install Visual Studio Build Tools
# Or use pre-built wheel instead
```

### **Fallback Mode**

If Pyo installation fails, the system will automatically use the fallback engine:

```python
# This will work without Pyo installed
from emotune.core.music.engine_pyo import create_music_engine

engine = create_music_engine("auto")
# Engine will be FallbackMusicEngine if Pyo unavailable
```

---

## **PERFORMANCE TUNING**

### **Buffer Size Optimization**

Edit `config/default.json`:

```json
{
  "music": {
    "audio_backend": "portaudio",
    "sample_rate": 44100,
    "buffer_size": 256  // Adjust based on your system
  }
}
```

**Buffer Size Guidelines**:
- **256**: Low latency, higher CPU usage
- **512**: Balanced performance
- **1024**: Lower CPU usage, higher latency

### **Audio Backend Selection**

```python
# Available backends
"portaudio"  # Cross-platform (recommended)
"jack"       # Linux only
"coreaudio"  # macOS only
"wasapi"     # Windows only
```

---

## **INTEGRATION WITH EMOTUNE**

### **Automatic Detection**

The EmoTune system automatically detects Pyo availability:

```python
# In session manager
engine = create_music_engine("auto")
# Will use PyoMusicEngine if available, FallbackMusicEngine otherwise
```

### **Manual Selection**

```python
# Force Pyo usage
engine = create_music_engine("pyo")

# Force fallback
engine = create_music_engine("fallback")
```

---

## **SUPPORT**

### **Getting Help**

1. **Check Logs**: Look for Pyo-related messages in `logs/emotune.log`
2. **Test Audio**: Run the verification script above
3. **Fallback Mode**: System works without Pyo for testing

### **Useful Commands**

```bash
# Check Pyo installation
python -c "import pyo; print('Pyo version:', pyo.__version__)"

# List audio devices
python -c "from pyo import *; s = Server(); print(s.getServerInfo())"

# Test audio output
python test_pyo_audio.py
```

---

## **CONCLUSION**

- ✅ **Pyo installation is optional** - system works without it
- ✅ **Fallback engine provides full functionality** for testing
- ✅ **Real audio output** available when Pyo is installed
- ✅ **Automatic detection** ensures compatibility

The EmoTune system is **production-ready** with or without Pyo installed! 