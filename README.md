# Predicting Focus vs. Distraction Based on Device Usage Behavior

## Files
- `focus_dataset.csv`: dataset with 300 samples
- `main.py`: training and evaluation script
- `final_report.pdf`: project report

## Dataset description
Each row represents one one-minute window of device usage behavior.

### Features
- `app_time`: time spent in the active app (seconds)
- `app_switch`: number of app switches
- `scroll_count`: number of scroll actions
- `keystrokes`: keyboard input count
- `notifications`: number of notifications received
- `time_period`: 1 = morning, 2 = afternoon, 3 = evening
- `label`: 1 = focused, 0 = distracted

## Note
This dataset is self-constructed and simulated for course project use.
