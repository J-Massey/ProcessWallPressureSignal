import numpy as np
import pandas as pd

# -------------------------------
# Configuration
# -------------------------------
n_tones = 12
run_length_s = 10  # seconds per tone

# Frequency bands [Hz]
bands = {
    "Atmospheric (0 psig)": {
        "HR": (700, 3000),
        "OP": (20, 200)
    },
    "50 psig": {
        "HR": (700, 3000),
        "OP": (50, 500)
    },
    "100 psig": {
        "HR": (700, 3000),
        "OP": (100, 800)
    }
}

# -------------------------------
# Build table
# -------------------------------
records = []
for pressure, band in bands.items():
    hr_freqs = np.round(np.geomspace(*band["HR"], num=n_tones), 1)
    op_freqs = np.round(np.geomspace(*band["OP"], num=n_tones), 1)
    records.append({
        "Pressure": pressure,
        "HR band [Hz]": f"{band['HR'][0]}–{band['HR'][1]}",
        "OP band [Hz]": f"{band['OP'][0]}–{band['OP'][1]}",
        "HR tones [Hz]": ", ".join(map(str, hr_freqs)),
        "OP tones [Hz]": ", ".join(map(str, op_freqs)),
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
