#!/usr/bin/env python3
"""
ACRL Setup Script
Automates the installation and setup process for ACRL training
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("✅ Python version is compatible for training")
        return True
    else:
        print("❌ Python 3.10+ is required for training")
        return False

def install_requirements():
    """Install required Python packages"""
    print("\n📦 Installing Python requirements...")
    
    requirements_file = Path("resource-repos/ACRL/standalone/requirements.txt")
    if not requirements_file.exists():
        print("❌ Requirements file not found")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_ac_installation():
    """Check if Assetto Corsa is installed"""
    print("\n🎮 Checking Assetto Corsa installation...")
    
    # Common AC installation paths
    ac_paths = [
        "C:/Program Files (x86)/Steam/steamapps/common/assettocorsa",
        "C:/Program Files/Steam/steamapps/common/assettocorsa",
        "D:/Steam/steamapps/common/assettocorsa"
    ]
    
    for path in ac_paths:
        if os.path.exists(path):
            print(f"✅ Assetto Corsa found at: {path}")
            return path
    
    print("❌ Assetto Corsa not found in common locations")
    print("Please install Assetto Corsa from Steam")
    return None

def copy_acrl_plugin(ac_path):
    """Copy ACRL plugin to Assetto Corsa"""
    print("\n🔌 Installing ACRL plugin...")
    
    source = Path("resource-repos/ACRL/ACRL")
    destination = Path(ac_path) / "apps" / "python" / "ACRL"
    
    if not source.exists():
        print("❌ ACRL plugin source not found")
        return False
    
    try:
        # Create destination directory
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy plugin files
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        
        print(f"✅ ACRL plugin installed to: {destination}")
        return True
    except Exception as e:
        print(f"❌ Failed to install plugin: {e}")
        return False

def create_training_script():
    """Create a convenient training script"""
    print("\n📝 Creating training script...")
    
    script_content = '''#!/usr/bin/env python3
"""
Quick ACRL Training Script
Run this to start training immediately
"""

import os
import sys

# Add ACRL standalone to path
acrl_path = os.path.join(os.path.dirname(__file__), "resource-repos/ACRL/standalone")
sys.path.insert(0, acrl_path)

# Import and run main
from main import main

if __name__ == "__main__":
    print("🚗 Starting ACRL Training...")
    print("Make sure Assetto Corsa is running with ACRL plugin enabled!")
    main()
'''
    
    script_path = Path("start_acrl_training.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print(f"✅ Training script created: {script_path}")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎯 NEXT STEPS TO START TRAINING")
    print("="*60)
    print("""
1. 🎮 Launch Assetto Corsa
   - Go to Settings → General tab
   - Enable the "ACRL" app

2. 🏁 Create a practice session:
   - Track: Silverstone 1967
   - Car: Ferrari 458 GT2
   - AI Opponents: 0
   - Framerate: 30 FPS
   - Enable automatic gear switching (Ctrl+G)

3. 🚗 Start training:
   - Run: python start_acrl_training.py
   - Or: cd resource-repos/ACRL/standalone && python main.py

4. ▶️ Begin training:
   - Wait for car to spawn on track
   - Click "Start Training" in ACRL app window
   - Watch the autonomous driving begin!

📚 For detailed instructions, see: resource-repos/findings/training-steps/ACRL_Training_Guide.md
""")

def main():
    """Main setup function"""
    print("🚗 ACRL Setup Script")
    print("="*40)
    
    # Check current directory
    if not Path("resource-repos").exists():
        print("❌ Please run this script from the project root directory")
        return False
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check AC installation
    ac_path = check_ac_installation()
    if ac_path:
        # Install plugin
        copy_acrl_plugin(ac_path)
    
    # Create training script
    create_training_script()
    
    # Print next steps
    print_next_steps()
    
    print("\n✅ Setup complete! Follow the next steps above to start training.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 