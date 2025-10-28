import numpy as np
import pandas as pd

# -------------------------------
# Configuration
# -------------------------------
n_tones = 16
run_length_s = 20  # seconds per tone

# Frequency bands [Hz]
bands = {
    "Atmospheric (0 psig)": {
        "HR": (700, 3000),
        "OP": (50, 200)
    },
    # "50 psig": {
    #     "HR": (700, 3000),
    #     "OP": (50, 500)
    # },
    # "100 psig": {
    #     "HR": (700, 3000),
    #     "OP": (50, 800)
    # }
}

# -------------------------------
# Build table
# -------------------------------
records = []
for pressure, band in bands.items():
    hr_freqs = np.round(np.geomspace(50, 2500, num=n_tones), 0)
    records.append({
        "Band [Hz]": r"50--2,500",
        "HR tones [Hz]": ", ".join(map(str, hr_freqs)),
        "Run length [s]": run_length_s
    })

# -------------------------------
# Display / Save
# -------------------------------
df = pd.DataFrame(records)
pd.set_option("display.max_colwidth", None)
print(df.to_markdown(index=False))  # pretty Markdown table

# Optionally save
df.to_csv("tonal_calibration_plan.csv", index=False)
