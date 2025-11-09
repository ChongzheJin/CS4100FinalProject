# Preventing MacBook Sleep During Training

## Quick Solution (Terminal Command)

Run this command to prevent sleep while plugged in:

```bash
caffeinate -d
```

This prevents the display from sleeping. Press `Ctrl+C` to stop it when training is done.

## More Options

### Prevent sleep completely (recommended for training):
```bash
caffeinate -d -i -m -s
```
- `-d`: Prevents display from sleeping
- `-i`: Prevents system from idle sleeping
- `-m`: Prevents disk from idle sleeping
- `-s`: Prevents system from sleeping (only works when plugged in)

### Prevent sleep for a specific duration:
```bash
caffeinate -d -i -m -s -t 36000  # 10 hours (36000 seconds)
```

### Run training with caffeinate:
```bash
caffeinate -d -i -m -s python src/train_agent1.py
```

## System Settings (Alternative)

1. **System Settings** → **Battery** (or **Energy Saver** on older macOS)
2. Set "Turn display off after" to **Never** (when plugged in)
3. Uncheck "Prevent automatic sleeping when display is off" (or keep it checked)

## Best Practice

**Recommended approach**: Run training with caffeinate:
```bash
caffeinate -d -i -m -s python src/train_agent1.py
```

This ensures your Mac won't sleep even if the display turns off, and training will continue uninterrupted.

## Check if it's working

You can verify caffeinate is active:
- The terminal will show it's running
- Your Mac won't sleep even if you close the lid (when plugged in)
- Training progress will continue

## When Training Finishes

Press `Ctrl+C` in the terminal to stop caffeinate, or it will stop automatically when the Python script finishes.

