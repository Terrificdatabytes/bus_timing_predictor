import pandas as pd
import numpy as np
from datetime import timedelta

# --- Configuration ---
NUM_ROWS = 10000
START_DATE = pd.to_datetime('2024-01-01 06:00:00')

# Define realistic ranges for features
SCHEDULED_TIME_MIN = 15  # 15 minutes
SCHEDULED_TIME_MAX = 70  # 70 minutes
PASSENGER_MIN = 5
PASSENGER_MAX = 60

# --- Data Generation ---

# 1. Generate Timestamps (uniformly distributed over a time span)
# Generate a series of random time offsets up to 1 year
time_offsets = [timedelta(days=i // 30, hours=np.random.randint(6, 23), minutes=np.random.randint(0, 60)) 
                for i in range(NUM_ROWS)]
timestamps = START_DATE + pd.Series(time_offsets)
timestamps = timestamps.sort_values().reset_index(drop=True) # Ensure they are sorted

# 2. Generate Scheduled Arrival Times
scheduled_arrivals = np.random.randint(SCHEDULED_TIME_MIN, SCHEDULED_TIME_MAX + 1, NUM_ROWS)

# 3. Generate Passenger Count (as an integer)
passenger_counts = np.random.randint(PASSENGER_MIN, PASSENGER_MAX + 1, NUM_ROWS)

# 4. Generate Actual Arrival Times (Target Variable)
# Simulate a realistic scenario: Actual arrival is scheduled time + some delay/early arrival (noise)
# Introduce two effects for more realistic variance:
#   a. Base noise (small random variation)
#   b. Effect of passengers (more passengers -> slightly more delay)

# Calculate Features for Noise Calculation
hours = timestamps.dt.hour

# a. Base Noise: Small random value, slightly higher during peak hours (e.g., 7-9 and 16-18)
base_noise = np.random.normal(0, 1.5, NUM_ROWS)
is_peak_morning = (hours >= 7) & (hours <= 9)
is_peak_evening = (hours >= 16) & (hours <= 18)
base_noise[is_peak_morning] += np.random.normal(1.0, 0.5, is_peak_morning.sum())
base_noise[is_peak_evening] += np.random.normal(0.8, 0.4, is_peak_evening.sum())

# b. Passenger Effect: Small positive delay factor proportional to passenger count
passenger_effect = (passenger_counts / PASSENGER_MAX) * np.random.uniform(0, 2, NUM_ROWS)

# Calculate Actual Arrival: Scheduled + Noise + Passenger Effect
actual_arrivals = scheduled_arrivals + base_noise + passenger_effect
actual_arrivals = np.clip(actual_arrivals, 5, 100) # Keep values reasonable and positive

# --- Create DataFrame and Export ---
data = pd.DataFrame({
    'timestamp': timestamps.dt.strftime('%Y-%m-%d %H:%M:%S'),
    'actual_arrival': actual_arrivals.round(2),
    'scheduled_arrival': scheduled_arrivals,
    'passenger_count': passenger_counts
})

# Save to CSV
csv_filename = 'bus_timing_data_10k.csv'
data.to_csv(csv_filename, index=False)

print(f"Successfully generated {NUM_ROWS} rows of data and saved to {csv_filename}")