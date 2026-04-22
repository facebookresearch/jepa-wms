import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_path = "logs/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save/simu_env_planning"

data = []
for dir_name in os.listdir(base_path):
    if not dir_name.startswith("cemgd_lri"):
        continue
    
    parts = dir_name.split("_")
    lri_str = parts[1].replace("lri", "").replace("p", ".")
    dec_str = parts[2].replace("dec", "").replace("p", ".")
    
    lri = float(lri_str)
    dec = float(dec_str)
    
    csv_path = os.path.join(base_path, dir_name, "eval.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            success = df['episode_success'].iloc[0]
            time_val = df['total_time'].iloc[0] / 20.0
            data.append({"lri": lri, "dec": dec, "success": success, "time": time_val})

df_res = pd.DataFrame(data)

pivot_success = df_res.pivot(index="dec", columns="lri", values="success")
pivot_time = df_res.pivot(index="dec", columns="lri", values="time")

# Sort ascending
pivot_success = pivot_success.sort_index(ascending=False)
pivot_time = pivot_time.sort_index(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(pivot_success, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[0])
axes[0].set_title("Episode Success")
axes[0].set_xlabel("Learning Rate (lri)")
axes[0].set_ylabel("Decay (dec)")

sns.heatmap(pivot_time, annot=True, fmt=".1f", cmap="Reds", ax=axes[1])
axes[1].set_title("Total Time / 20")
axes[1].set_xlabel("Learning Rate (lri)")
axes[1].set_ylabel("Decay (dec)")

plt.tight_layout()
out_img = "sweep_results.png"
plt.savefig(out_img)
print(f"Plot saved to {out_img}")
