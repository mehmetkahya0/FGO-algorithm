"""Tüm deney parametreleri tek yerde."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 7

    # --- Dünya ---
    world_size: float = 1000.0
    map_res: float = 10.0            # önsel manyetik / batimetri harita çözünürlüğü [m]
    plan_res: float = 10.0           # planlama ızgarası [m]
    thermocline_depth: float = 45.0  # termoklin derinliği [m]
    min_seafloor_depth: float = 30.0  # daha sığ bölgeler engel kabul edilir

    # --- Zaman ---
    dt: float = 0.5                  # simülasyon adımı [s]
    keyframe_every: int = 4          # 2 s'de bir anahtar kare (faktör grafiği düğümü)
    max_time: float = 2800.0

    # --- Araç / kontrol ---
    cruise_speed: float = 1.8
    stealth_speed: float = 1.0
    max_yaw_rate: float = 0.12
    yaw_gain: float = 0.8
    altitude_cmd: float = 8.0        # deniz tabanını takip irtifası [m]
    max_vz: float = 0.6
    lookahead: float = 30.0
    goal_tolerance: float = 15.0
    start: tuple = (60.0, 60.0)
    goals: tuple = ((420.0, 380.0), (820.0, 520.0), (190.0, 830.0), (920.0, 930.0))

    # --- Sensörler ---
    gyro_noise: float = 0.002            # rad/s (örnek başı)
    gyro_bias0: float = 0.0015           # rad/s başlangıç sapması (bilinmiyor)
    gyro_bias_rw: float = 2e-5           # rad/s/sqrt(s)
    dvl_noise: float = 0.02              # m/s
    dvl_scale_error: float = 0.006       # modellenmemiş ölçek hatası
    dvl_model_sigma: float = 0.25        # DVL kesintisinde pervane modelinin belirsizliği
    dvl_sigma_model: float = 0.04        # FGO'nun DVL için kullandığı (şişirilmiş) gürültü
    pressure_noise: float = 0.05         # m
    compass_noise_deg: float = 2.0
    compass_misalign_deg: float = 1.5    # modellenmemiş montaj hatası
    compass_anomaly_coupling: float = 4e-4  # rad/nT  (manyetik anomaliler pusulayı bozar)
    mag_noise: float = 2.0               # nT
    mag_map_noise: float = 0.5           # nT (önsel harita hatası)
    sonar_range: float = 80.0
    sonar_fov_deg: float = 65.0
    sonar_pd: float = 0.85
    sonar_range_sigma: float = 0.4
    sonar_range_sigma_rel: float = 0.005
    sonar_bearing_sigma_deg: float = 1.2
    sonar_elev_sigma_deg: float = 2.0
    sonar_false_alarm_rate: float = 0.08  # tarama başına yanlış alarm beklentisi
    sonar_multipath_prob: float = 0.05    # çoklu-yol yankısı (kaba aykırı ölçüm) olasılığı
    optical_sigma: float = 0.12
    mag_feature_range: float = 12.0
    hydrophone_bearing_sigma_deg: float = 4.0

    # --- FGO ---
    opt_every: int = 5
    opt_iters: int = 3
    final_iters: int = 25
    huber_k: float = 2.0
    cauchy_k: float = 2.5
    mag_kernel: str = "huber"
    lm_kernel: str = "cauchy"
    compass_sigma_model_deg: float = 3.0
    mag_sigma_model: float = 3.0
    assoc_gate: float = 7.0

    # --- Tehdit / gizlilik ---
    threat_prior_p: float = 2e-4
    threat_decay_tau: float = 900.0
    nominal_threat_radius: float = 140.0
    nominal_threat_rate: float = 0.01

    # --- Planlama ---
    w_risk: float = 1500.0
    w_info: float = 0.3
    replan_every: int = 15
    stealth_hazard_threshold: float = 0.002
