# ACRL Quick Reference Card

## 🚀 Quick Start Commands

### 1. Automated Setup
```bash
python resource-repos/findings/training-steps/setup_acrl.py
```

### 2. Manual Setup
```bash
# Install dependencies
cd resource-repos/ACRL/standalone
pip install -r requirements.txt

# Start training
python main.py
```

### 3. Quick Training Start
```bash
python start_acrl_training.py
```

## 🎮 Assetto Corsa Setup

### Required Session Settings
- **Track**: Silverstone 1967
- **Car**: Ferrari 458 GT2
- **Mode**: Practice
- **AI**: 0 opponents
- **Framerate**: 30 FPS
- **Gear**: Automatic (Ctrl+G)

### Plugin Installation
1. Copy `resource-repos/ACRL/ACRL/` to `[AC_Install]/apps/python/`
2. Enable "ACRL" app in AC Settings → General

## 📊 Training Monitoring

### Expected Timeline
- **Episodes 1-100**: Random exploration, crashes
- **Episodes 100-500**: Basic track following
- **Episodes 500-1000**: Consistent laps
- **Episodes 1000+**: Optimized racing

### Success Indicators
- ✅ Car stays on track consistently
- ✅ Lap times improving
- ✅ Reduced crash frequency
- ✅ Stable racing line

## 🔧 Troubleshooting

### Common Issues
| Problem | Solution |
|---------|----------|
| Connection failed | Enable ACRL plugin in AC |
| Training not starting | Check car spawned, auto-gear enabled |
| Poor performance | Reduce AC to 30 FPS |
| Python errors | Use Python 3.10+ for training |

### Performance Tips
- Use dedicated GPU for PyTorch
- Close unnecessary applications
- Monitor system resources
- Use SSD for data access

## 📁 Key Files

```
ACRL/
├── standalone/main.py          # Main training script
├── standalone/requirements.txt # Python dependencies
├── ACRL/ACRL.py               # AC plugin
└── track_data/spline_points.csv # Track reference
```

## 🎯 Training Parameters

### Default Settings
- **Algorithm**: SAC (Soft Actor-Critic)
- **Episodes**: 10,000
- **Learning Rate**: 1e-3
- **Batch Size**: 32
- **Max Speed**: 180 km/h

### Car Configuration
- **Steering Range**: [-270°, 270°]
- **Track**: Silverstone 1967
- **Car**: Ferrari 458 GT2

## 📚 Additional Resources

- **Full Guide**: `ACRL_Training_Guide.md`
- **Original Repo**: https://github.com/jurredr/ACRL
- **SAC Paper**: https://arxiv.org/abs/1801.01290

## 🚨 Emergency Stop

If training goes wrong:
1. Close Assetto Corsa
2. Stop Python script (Ctrl+C)
3. Restart AC session
4. Re-run training script 