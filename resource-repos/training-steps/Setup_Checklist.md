# ACRL Setup Checklist

## ✅ Pre-Training Checklist

### System Requirements
- [ ] **Assetto Corsa** installed from Steam
- [ ] **Content Manager** installed (free extension)
- [ ] **Python 3.10+** installed for training
- [ ] **Python 3.3.5** installed for AC plugin (optional, can use newer)
- [ ] **Microsoft Visual C++ 2015-2019 Redistributable** (Windows)
- [ ] **Microsoft MPI v10.1.3** (Windows)
- [ ] **GPU drivers** updated (for PyTorch acceleration)

### Software Installation
- [ ] **ACRL plugin** copied to AC installation
- [ ] **Python dependencies** installed (`pip install -r requirements.txt`)
- [ ] **ACRL app** enabled in Assetto Corsa settings
- [ ] **Training script** created (`start_acrl_training.py`)

### Assetto Corsa Configuration
- [ ] **Practice session** created with correct settings:
  - [ ] Track: Silverstone 1967
  - [ ] Car: Ferrari 458 GT2
  - [ ] AI Opponents: 0
  - [ ] Tires: Default
  - [ ] Penalties: ON
  - [ ] Ideal conditions: ON
  - [ ] Framerate: 30 FPS
  - [ ] Controls: Gamepad
  - [ ] Speed sensitivity: 0
  - [ ] Steering speed: 100%
  - [ ] Steering gamma: 100%
  - [ ] Steering filter: 0%
  - [ ] Automatic gear switching: Enabled (Ctrl+G)

### Network & Communication
- [ ] **Firewall** allows Python/AC communication
- [ ] **Socket ports** available (default: 2345, 2346, 2347)
- [ ] **No antivirus** blocking communication
- [ ] **Network adapter** properly configured

### Performance Optimization
- [ ] **Unnecessary applications** closed
- [ ] **System resources** monitored
- [ ] **SSD** available for faster data access
- [ ] **GPU** dedicated for PyTorch (if available)
- [ ] **Stable 30 FPS** maintained in Assetto Corsa

## 🚀 Training Start Checklist

### Before Starting Training
- [ ] **Assetto Corsa** launched and session loaded
- [ ] **Car spawned** on track
- [ ] **Automatic gear switching** enabled
- [ ] **ACRL app window** visible in AC
- [ ] **Training script** ready to run
- [ ] **Console/terminal** open for monitoring

### Training Script Execution
- [ ] **Navigate** to correct directory: `resource-repos/ACRL/standalone`
- [ ] **Run** training script: `python main.py`
- [ ] **Enter** experiment name when prompted
- [ ] **Click** "Start Training" in ACRL app window
- [ ] **Monitor** console output for errors

### Expected Initial Behavior
- [ ] **Socket connection** established successfully
- [ ] **Car begins** autonomous movement
- [ ] **Random actions** initially (exploration phase)
- [ ] **Episode counter** incrementing
- [ ] **Loss values** being logged
- [ ] **No immediate crashes** (car stays on track briefly)

## 📊 Training Monitoring Checklist

### During Training (First 100 Episodes)
- [ ] **Episode counter** increasing
- [ ] **Loss values** fluctuating (normal)
- [ ] **Car occasionally** staying on track
- [ ] **No connection errors** in console
- [ ] **AC maintaining** 30 FPS
- [ ] **System resources** stable

### During Training (Episodes 100-500)
- [ ] **Crash frequency** decreasing
- [ ] **Lap completion** becoming more common
- [ ] **Racing line** becoming more consistent
- [ ] **Loss values** trending downward
- [ ] **Episode duration** increasing

### During Training (Episodes 500+)
- [ ] **Consistent lap completion** (>80% success rate)
- [ ] **Stable racing line** following
- [ ] **Improved lap times** over baseline
- [ ] **Minimal crashes** (<10% frequency)
- [ ] **Smooth steering** and acceleration

## 🔧 Troubleshooting Checklist

### If Training Won't Start
- [ ] **ACRL plugin** properly installed
- [ ] **ACRL app** enabled in AC settings
- [ ] **Python dependencies** installed correctly
- [ ] **Correct session** loaded in AC
- [ ] **Car spawned** on track
- [ ] **Automatic gear switching** enabled

### If Connection Fails
- [ ] **Firewall** allows Python/AC communication
- [ ] **Socket ports** not blocked
- [ ] **ACRL plugin** enabled
- [ ] **Session** properly loaded
- [ ] **No antivirus** interference

### If Performance is Poor
- [ ] **AC framerate** set to 30 FPS
- [ ] **Unnecessary apps** closed
- [ ] **GPU drivers** updated
- [ ] **System resources** adequate
- [ ] **SSD** used for data access

### If Training is Unstable
- [ ] **Checkpoint files** being saved
- [ ] **Loss values** reasonable
- [ ] **Episode duration** consistent
- [ ] **No memory leaks** in system
- [ ] **Network connection** stable

## 🎯 Success Criteria Checklist

### After 1000 Episodes
- [ ] **Consistent lap completion** (>90% success rate)
- [ ] **Stable racing line** following
- [ ] **Improved lap times** over random baseline
- [ ] **Minimal crashes** (<5% frequency)
- [ ] **Smooth control** inputs
- [ ] **Checkpoint files** saved regularly

### Model Evaluation
- [ ] **Training checkpoints** available
- [ ] **Loss convergence** achieved
- [ ] **Policy improvement** evident
- [ ] **Transfer learning** possible
- [ ] **Performance metrics** logged

## 📚 Documentation Checklist

### Files Created
- [ ] **Training logs** saved
- [ ] **Checkpoint files** created
- [ ] **Performance metrics** recorded
- [ ] **Configuration** documented
- [ ] **Results** summarized

### Next Steps Planned
- [ ] **Model evaluation** on different tracks
- [ ] **Hyperparameter tuning** identified
- [ ] **Transfer learning** experiments planned
- [ ] **Performance optimization** opportunities noted
- [ ] **Advanced features** to implement

---

**Note**: Check off each item as you complete it. If any item cannot be completed, refer to the troubleshooting section in the main guide. 