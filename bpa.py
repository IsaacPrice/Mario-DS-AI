from datetime import datetime, timedelta

# Current date and time
now = datetime.now()

# Target date and time
target_date = datetime(year=2024, month=3, day=10, hour=8, minute=0, second=0)

# Calculate difference
time_until_target = target_date - now

time_until_target.days, time_until_target.seconds // 3600, (time_until_target.seconds // 60) % 60

print(f"Days: {time_until_target.days}, Hours: {time_until_target.seconds // 3600}, Minutes: {(time_until_target.seconds // 60) % 60}")