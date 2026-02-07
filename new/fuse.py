import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# CONFIGURATION
# ==============================
# 1. INPUT PATHS (Update these if your folder names differ slightly)
PATH_POS      = r"D:\new2\venv\rPPG-Toolbox\new\final_pos_results-"           
PATH_PHYSNET  = r"D:\new2\venv\rPPG-Toolbox\new\final_physnet_graphs_full-"       
PATH_EFFPHYS  = r"D:\new2\venv\rPPG-Toolbox\new\final_efficientphys_graphs_full-" 

# 2. OUTPUT FOLDER
OUTPUT_FOLDER = r"./FINAL_WEIGHTED_RESULTS"

# 3. WEIGHTS (Must sum to 1.0)
W_EFFPHYS = 0.60
W_POS     = 0.20
W_PHYSNET = 0.20

# ==============================
# HELPER: RECURSIVE FILE MAPPER
# ==============================
def map_files(root_folder):
    """
    Recursively finds all .csv files in a folder and its subfolders.
    Returns: { "clean_video_id": "full/path/to/file.csv" }
    """
    file_map = {}
    print(f"📂 Scanning: {root_folder}...", end="")
    
    if not os.path.exists(root_folder):
        print(" ❌ FOLDER NOT FOUND!")
        return file_map

    count = 0
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv") and "ensemble" not in file:
                # 1. Clean the ID to ensure matches across folders
                # Remove common suffixes like _filtered, _trace, _clean
                vid_id = os.path.splitext(file)[0]
                vid_id = vid_id.replace("_filtered", "").replace("_trace", "").replace("_clean", "")
                
                # 2. Map ID to Full Path
                file_map[vid_id] = os.path.join(root, file)
                count += 1
                
    print(f" Found {count} files.")
    return file_map

# ==============================
# CORE LOGIC: WEIGHTED MEAN
# ==============================
def process_fusion(vid_id, p_pos, p_phy, p_eff):
    try:
        # 1. Load Data
        df_pos = pd.read_csv(p_pos)
        df_phy = pd.read_csv(p_phy)
        df_eff = pd.read_csv(p_eff)
        
        # 2. Extract Signal (Assume last column is the data)
        # Force numeric to handle errors, fill NaNs with 0
        s_pos = pd.to_numeric(df_pos.iloc[:, -1], errors='coerce').fillna(0).values
        s_phy = pd.to_numeric(df_phy.iloc[:, -1], errors='coerce').fillna(0).values
        s_eff = pd.to_numeric(df_eff.iloc[:, -1], errors='coerce').fillna(0).values
        
        # 3. Sync Length (Trim to shortest video)
        min_len = min(len(s_pos), len(s_phy), len(s_eff))
        if min_len < 30: return False # Skip empty/broken files
        
        s_pos = s_pos[:min_len]
        s_phy = s_phy[:min_len]
        s_eff = s_eff[:min_len]
        
        # Get Time Axis from POS file
        time_axis = df_pos.iloc[:min_len, 0].values

        # 4. CALCULATE WEIGHTED MEAN
        # Formula: (V1*0.2) + (V2*0.2) + (V3*0.6)
        weighted_signal = (s_pos * W_POS) + (s_phy * W_PHYSNET) + (s_eff * W_EFFPHYS)
        
        # 5. SAVE CSV
        out_csv = os.path.join(OUTPUT_FOLDER, f"{vid_id}_weighted.csv")
        pd.DataFrame({
            'Time_Sec': time_axis, 
            'Weighted_Value': weighted_signal
        }).to_csv(out_csv, index=False)
        
        # 6. SAVE GRAPH (Visual Check)
        plt.figure(figsize=(10, 5))
        plt.plot(time_axis, weighted_signal, color='#dc2626', linewidth=1.5, label='Fused Signal')
        
        avg_val = np.mean(weighted_signal)
        plt.axhline(y=avg_val, color='#2563eb', linestyle='--', label=f'Mean: {avg_val:.1f}')
        
        plt.title(f"Video: {vid_id} | Weighted Mean: {avg_val:.1f}")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"{vid_id}_graph.png"), dpi=100)
        plt.close()
        
        return True

    except Exception as e:
        print(f"❌ Error processing {vid_id}: {e}")
        return False

# ==============================
# MAIN LOOP
# ==============================
def main():
    print("🚀 Starting Final Weighted Fusion (60% EffPhys / 20% POS / 20% PhysNet)...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 1. Map Files
    map_pos = map_files(PATH_POS)
    map_phy = map_files(PATH_PHYSNET)
    map_eff = map_files(PATH_EFFPHYS)
    
    # 2. Find Common IDs (Intersection)
    common_ids = set(map_pos.keys()) & set(map_phy.keys()) & set(map_eff.keys())
    
    if not common_ids:
        print("\n❌ CRITICAL: No matching video IDs found.")
        print("   Check your filenames! They must share a common 'core' name.")
        return

    print(f"\n✅ Found {len(common_ids)} matching videos. Processing...\n")
    
    count = 0
    for i, vid_id in enumerate(common_ids):
        print(f"[{i+1}/{len(common_ids)}] Fusing: {vid_id}...", end="")
        
        if process_fusion(vid_id, map_pos[vid_id], map_phy[vid_id], map_eff[vid_id]):
            print(" Done.")
            count += 1
        else:
            print(" Failed.")

    print("\n" + "="*50)
    print(f"🎉 DONE! Processed {count} videos.")
    print(f"📂 Results saved in: {OUTPUT_FOLDER}")
    print("="*50)

if __name__ == "__main__":
    main()