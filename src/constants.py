import os
import numpy as np

DATA_DIR = os.environ.get("TPOT_DATA_DIR", "t-pot_data_analysis")
BENIGN_FILE = os.path.join(DATA_DIR, "smartHomeSandbox", "sandbox", "dataset", "network_traffic.csv")
MALICIOUS_FILE = os.path.join(DATA_DIR, "malicious_dataset", "network_traffic.csv")
SCENARIOS_DIR = os.path.join(DATA_DIR, "smartHomeSandbox", "sandbox", "dataset", "data_scenarios")

BENIGN_CONNECTION_FILE = os.path.join(DATA_DIR, "smartHomeSandbox", "sandbox", "dataset", "network_binary_connection.csv")
BENIGN_BINARY_INFO_FILE = os.path.join(DATA_DIR, "smartHomeSandbox", "sandbox", "dataset", "binary_information.csv")
MALICIOUS_CONNECTION_FILE = os.path.join(DATA_DIR, "malicious_dataset", "ip.csv")
MALICIOUS_BINARY_INFO_FILE = os.path.join(DATA_DIR, "malicious_dataset", "binary_information.csv")

IOTPOT_BINARY_INFO_FILE = os.path.join(DATA_DIR, "iotpot", "binary_information.csv")

"""
One client per physical IoT device (baseline model). Grouped by folder name in 
SCENARIOS_DIR (t-pot_data_analysis/smartHomeSandbox/sandbox/dataset/data_scenarios/),
"idle" and "routine" are left out
"""
DEVICE_SCENARIOS = {
    "laptop":         ["laptop_bridge_control", "laptop_light_control", "laptop_plug_control"],
    "mobile":         ["mobile_bridge_control", 
                       "mobile_camera_view", 
                       "mobile_light_control", 
                       "mobile_panel_call", 
                       "mobile_plug_control", 
                       "mobile_wifi_toggle"
                      ],
    "panel":          ["panel_bridge_control", 
                       "panel_camera_view", 
                       "panel_light_control", 
                       "panel_plug_control", 
                       "panel_power"
                      ],
    "hue_bridge":     ["bridge_power", "bridge_update", "hue_bridge_control"],
    "hue_lamp":       ["hue_light_control", "connect_lamp", "lamp_power"],
    "camera":         ["camera_power"],
    "socket":         ["socket_connect", "socket_power"],
    "ha_speaker":     ["HA_login_attempts", 
                       "HA_update", 
                       "ssh_login_attempts", 
                       "speaker_power", 
                       "speaker_voice_query"
                      ], # Home Assisstant and voice assisst are on one device
    "diy_hue":        ["diyHue_login_attempts"],
    "motion_sensor":  ["sensor_automation_trigger"]
}
DEVICE_NAMES = list(DEVICE_SCENARIOS.keys())  # partition_id -> device name

rng = np.random.default_rng(seed=42)

# binary_information.csv columns split into the ones that are already numeric vs. the ones that need a text hash.
BINFO_NUMERIC_COLS = [
    "count",
    "obfuscation_Size(Bytes)",
    "obfuscation_Strings",
    "obfuscation_Strings_per_kB",
    "obfuscation_Entropy"
]

# Columns from binary_information.csv excluded as features
BINFO_EXCLUDED_COLS = [
    "c2_ports",
    "anti_forensic",
    "anti_debugging",
    "anti_sandbox",
    "malware_name"
]
BINFO_TEXT_COLS = [
    "file_type",
    "cpu_arch",
    "linkage",
    "persistence"
]

FEATURES = (
    ["count", "url"]
    + [f"binfo_{c}" for c in BINFO_NUMERIC_COLS]
    + [f"binfo_{c}" for c in BINFO_TEXT_COLS]
)
INPUT_DIM = len(FEATURES)
