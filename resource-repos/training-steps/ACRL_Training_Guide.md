# ACRL Training Guide: Autonomous Racing with SAC

## Overview
This guide will walk you through setting up and running the ACRL (Assetto Corsa Reinforcement Learning) project to train an autonomous driving agent using the SAC (Soft Actor-Critic) algorithm.

## Prerequisites

### 1. Assetto Corsa Installation
- Purchase and install [Assetto Corsa](https://store.steampowered.com/app/244210/Assetto_Corsa/) on Steam
- Download and install [Content Manager](https://assettocorsa.club/content-manager.html) (free extension)

### 2. System Requirements
- **Windows**: Microsoft Visual C++ 2015-2019 Redistributable
- **Windows**: Microsoft Messaging Passing Interface (MS-MPI) v10.1.3
- **Python 3.3.5**: For Assetto Corsa plugin compatibility
- **Python 3.10.11**: For standalone training (recommended)

## Step-by-Step Setup

### Step 1: Install Python Dependencies

```bash
# Navigate to the ACRL standalone directory
cd resource-repos/ACRL/standalone

# Install required packages
pip install -r requirements.txt
```

**Required packages:**
- cmake
- numpy
- torch
- keyboard
- vgamepad
- matplotlib
- gymnasium
- mpi4py
- joblib

### Step 2: Install Assetto Corsa Plugin

1. **Install Python 3.3.5** (required for AC plugin compatibility)
   - Download from: https://legacy.python.org/download/releases/3.3.5/
   - Install to a separate directory (e.g., `C:\Python33`)

2. **Copy ACRL plugin to Assetto Corsa**
   ```bash
   # Copy the ACRL folder to your AC installation
   cp -r resource-repos/ACRL/ACRL/ "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\python\"
   ```

3. **Enable the ACRL plugin in Assetto Corsa**
   - Launch Assetto Corsa
   - Go to Settings → General tab
   - Enable the "ACRL" app
   - Or enable through Content Manager settings

### Step 3: Configure Assetto Corsa Session

Create a new practice session with these exact settings:

**Track**: Silverstone 1967
**Car**: Ferrari 458 GT2
**Mode**: Practice
**AI Opponents**: 0
**Tires**: Default
**Settings**:
- Penalties: ON
- Ideal conditions: ON
- Framerate: 30 FPS
- Controls: Gamepad
- Speed sensitivity: 0
- Steering speed: 100%
- Steering gamma: 100%
- Steering filter: 0%
- Automatic gear switching: Enabled (Ctrl+G)

### Step 4: Start Training

1. **Launch Assetto Corsa session**
   - Start the configured practice session
   - Wait for car to spawn on track
   - Ensure automatic gear switching is enabled

2. **Start the training script**
   ```bash
   # Navigate to standalone directory
   cd resource-repos/ACRL/standalone
   
   # Run the main training script
   python main.py
   ```

3. **Begin training**
   - The script will prompt for experiment name
   - Enter a descriptive name (e.g., "silverstone_training_v1")
   - Click "Start Training" in the ACRL app window in Assetto Corsa
   - The car will begin autonomous driving and training

## Training Process

### What Happens During Training

1. **Environment Setup**: The SAC agent connects to Assetto Corsa via socket
2. **Data Collection**: Real-time telemetry data is collected from the car
3. **Policy Learning**: The agent learns optimal driving actions using SAC algorithm
4. **Continuous Improvement**: The model improves lap times over episodes

### Training Parameters

**Default Hyperparameters:**
- Gamma: 0.99 (discount factor)
- Learning rate: 1e-3
- Batch size: 32
- Episodes: 10,000
- Update frequency: Every 50 steps
- Replay buffer size: 1,000,000

**Car Configuration:**
- Max speed: 180.0 km/h
- Steering scale: [-270, 270] degrees

### Monitoring Training

**Console Output:**
- Episode progress
- Loss values
- Reward statistics
- Training metrics

**Expected Behavior:**
- Car starts with random actions
- Gradually learns to stay on track
- Improves lap times over time
- Eventually achieves consistent racing lines

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Ensure ACRL plugin is enabled in AC
   - Check firewall settings
   - Verify socket ports are available

2. **Python Version Conflicts**
   - Use Python 3.3.5 for AC plugin
   - Use Python 3.10+ for training script
   - Keep environments separate

3. **Training Not Starting**
   - Verify car is spawned on track
   - Check automatic gear switching is enabled
   - Ensure "Start Training" button is clicked in AC

4. **Poor Performance**
   - Reduce AC framerate to 30 FPS
   - Close unnecessary applications
   - Check GPU drivers are updated

### Performance Optimization

**For Better Training:**
- Use dedicated GPU for PyTorch
- Ensure stable 30 FPS in Assetto Corsa
- Monitor system resources during training
- Use SSD for faster data access

## Expected Results

### Training Timeline
- **Episodes 1-100**: Random exploration, frequent crashes
- **Episodes 100-500**: Basic track following, reduced crashes
- **Episodes 500-1000**: Consistent lap completion
- **Episodes 1000+**: Optimized racing lines, improved lap times

### Success Metrics
- Consistent lap completion (>90% success rate)
- Stable racing line following
- Improved lap times over baseline
- Reduced crash frequency

## Next Steps

### After Training
1. **Save Model**: Training checkpoints are automatically saved
2. **Evaluate Performance**: Test on different tracks/cars
3. **Fine-tune**: Adjust hyperparameters for better performance
4. **Transfer Learning**: Use pre-trained model for new scenarios

### Advanced Features
- **Multi-track Training**: Extend to other tracks
- **Different Cars**: Adapt to various vehicle types
- **Weather Conditions**: Train in different weather
- **Competitive Racing**: Add opponent cars

## File Structure

```
ACRL/
├── ACRL/                    # AC plugin files
│   ├── ACRL.py             # Main plugin script
│   └── ac_api/             # API interfaces
├── standalone/             # Training environment
│   ├── main.py             # Main training script
│   ├── sac/                # SAC algorithm implementation
│   ├── ac_environment.py   # Gym environment wrapper
│   └── requirements.txt    # Python dependencies
└── track_data/            # Track-specific data
    └── spline_points.csv   # Track reference points
```

## Support

For issues or questions:
- Check the original ACRL repository: https://github.com/jurredr/ACRL
- Review console error messages
- Verify all prerequisites are installed correctly
- Ensure Assetto Corsa and Content Manager are up to date 