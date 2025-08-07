#!/usr/bin/env python3
"""
Installation script for Pyo music engine
"""

import subprocess
import sys
import os

def install_pyo():
    """Install Pyo with proper error handling"""
    print("Installing Pyo music engine...")
    
    try:
        # Try to install Pyo
        result = subprocess.run([sys.executable, "-m", "pip", "install", "pyo"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Pyo installed successfully!")
            return True
        else:
            print(f"✗ Pyo installation failed: {result.stderr}")
            print("\nManual installation required:")
            print("1. Download Pyo from: https://pypi.org/project/pyo/#files")
            print("2. Choose the appropriate wheel for your Python version")
            print("3. Install with: pip install <downloaded_wheel_file>")
            return False
            
    except Exception as e:
        print(f"✗ Installation error: {e}")
        return False

def verify_installation():
    """Verify Pyo installation"""
    print("\nVerifying Pyo installation...")
    
    try:
        import pyo
        print("✓ Pyo imported successfully!")
        
        # Test basic functionality
        s = pyo.Server(audio="portaudio")
        print("✓ Pyo server created successfully!")
        s.shutdown()
        
        return True
        
    except ImportError:
        print("✗ Pyo not found. Installation may have failed.")
        return False
    except Exception as e:
        print(f"✗ Pyo verification failed: {e}")
        return False

def check_system_requirements():
    """Check system requirements for Pyo"""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 8:
        print("✓ Python version is compatible")
    else:
        print("⚠ Python 3.8+ recommended for best compatibility")
    
    # Check if we're on Windows
    if sys.platform.startswith('win'):
        print("✓ Windows detected - using PortAudio backend")
    elif sys.platform.startswith('linux'):
        print("✓ Linux detected - using PortAudio backend")
    elif sys.platform.startswith('darwin'):
        print("✓ macOS detected - using PortAudio backend")
    else:
        print("⚠ Unknown platform - audio may not work")
    
    return True

def main():
    """Main installation process"""
    print("Pyo Music Engine Installation")
    print("=" * 40)
    
    # Check system requirements
    check_system_requirements()
    
    # Install Pyo
    if install_pyo():
        # Verify installation
        if verify_installation():
            print("\n🎉 Pyo installation completed successfully!")
            print("The EmoTune system is now ready to use Pyo for music generation.")
            return True
        else:
            print("\n⚠ Pyo installation may have issues. Please check manually.")
            return False
    else:
        print("\n❌ Pyo installation failed. Please install manually.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 