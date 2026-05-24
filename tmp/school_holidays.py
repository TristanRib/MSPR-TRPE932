import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from vacances_scolaires_france import SchoolHolidayDates

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from transform import _CONSO_COL

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PARAMS = dict(
    n_estimators=978, learning_rate=0.046, max_depth=6,
    subsample=0.75, colsample_bytree=0.93, min_child_weight=5,
    gamma=4.0, reg_alpha=0.65, reg_lambda=2.65,
    random_state=42, n_jobs=-1,
)

# --- Données ---
df = pd.read_csv(DATA_DIR / "transformed_data.csv")
raw = pd.read_csv(DATA_DIR / "raw_data.csv", sep=";", encoding="utf-8-sig", low_memory=False)
raw = raw.sort_values("Date et Heure").reset_index(drop=True)
raw_dt = pd.to_datetime(raw["Date et Heure"], utc=True)
raw = raw[~raw_dt.duplicated(keep="first")].reset_index(drop=True)
raw_dt = pd.to_datetime(raw["Date et Heure"], utc=True)
mask_conso = pd.to_numeric(raw["Consommation (MW)"], errors="coerce").notna()
raw_dt_valid = raw_dt[mask_conso].reset_index(drop=True)
conso_hist = pd.Series(pd.to_numeric(raw["Consommation (MW)"], errors="coerce").values, index=raw_dt)

print("Calcul conso_mean_52w...")
idx = pd.DatetimeIndex(raw_dt_valid)
vals_52w = np.stack(
    [conso_hist.reindex(idx - pd.Timedelta(hours=168 * w)).to_numpy() for w in range(1, 53)], axis=1,
)
conso_mean_52w = np.nanmean(vals_52w, axis=1)[:len(df)]

# --- Construire le calendrier vacances scolaires ---
print("Construction calendrier vacances scolaires...")
d = SchoolHolidayDates()

# Récupérer toutes les dates de vacances 2011-2027
all_holidays: dict[date, dict] = {}
for yr in range(2011, 2028):
    try:
        all_holidays.update(d.holidays_for_year(yr))
    except Exception:
        pass

# Construire lookup : date -> {zone_a, zone_b, zone_c, nom}
holiday_lookup = {
    dt: {
        "A": info["vacances_zone_a"],
        "B": info["vacances_zone_b"],
        "C": info["vacances_zone_c"],
        "nom": info["nom_vacances"],
    }
    for dt, info in all_holidays.items()
}

# Mapping nom -> id ordinal (ordre calendaire)
NOM_ID = {
    "Vacances de Noël": 0,
    "Vacances d'hiver": 1,
    "Vacances de printemps": 2,
    "Vacances d'été": 3,
    "Vacances de la Toussaint": 4,
    "Pont de l'Ascension": 5,
}
CAP_DAYS = 14  # cap pour days_to/since

# Toutes les dates de vacances triées par zone
dates_sorted = sorted(holiday_lookup.keys())

def nearest_boundary(dt: date):
    """Retourne (days_to_start, days_since_end, in_holiday, nom, progress)."""
    if dt in holiday_lookup:
        # On est en vacances — trouver début et fin du bloc courant
        start = dt
        while (start - timedelta(days=1)) in holiday_lookup:
            start -= timedelta(days=1)
        end = dt
        while (end + timedelta(days=1)) in holiday_lookup:
            end += timedelta(days=1)
        length = (end - start).days + 1
        progress = (dt - start).days / max(length - 1, 1)
        nom = holiday_lookup[dt]["nom"]
        return 0, 0, True, nom, progress
    else:
        # Pas en vacances — chercher prochaines vacances et dernières vacances
        next_start = next((d for d in dates_sorted if d > dt), None)
        prev_end   = next((d for d in reversed(dates_sorted) if d < dt), None)
        days_to   = min((next_start - dt).days, CAP_DAYS) if next_start else CAP_DAYS
        days_since = min((dt - prev_end).days,  CAP_DAYS) if prev_end   else CAP_DAYS
        return days_to, days_since, False, "", 0.0

# --- Appliquer sur toutes les lignes ---
print("Application sur toutes les lignes...")
dt_paris = idx.tz_convert("Europe/Paris")
dates_paris = [ts.date() for ts in dt_paris]

rows = [nearest_boundary(dt) for dt in dates_paris]
days_to    = np.array([r[0] for r in rows], dtype=float)[:len(df)]
days_since = np.array([r[1] for r in rows], dtype=float)[:len(df)]
in_hol     = np.array([r[2] for r in rows], dtype=float)[:len(df)]
nom_arr    = [r[3] for r in rows][:len(df)]
progress   = np.array([r[4] for r in rows], dtype=float)[:len(df)]

hol_type = np.array([NOM_ID.get(n, -1) for n in nom_arr], dtype=float)

zone_a = np.array([holiday_lookup.get(dt, {}).get("A", False) for dt in dates_paris], dtype=float)[:len(df)]
zone_b = np.array([holiday_lookup.get(dt, {}).get("B", False) for dt in dates_paris], dtype=float)[:len(df)]
zone_c = np.array([holiday_lookup.get(dt, {}).get("C", False) for dt in dates_paris], dtype=float)[:len(df)]

print(f"  Jours de vacances (zone A) : {zone_a.sum():.0f} / {len(df)}")
print(f"  days_to_start mean         : {days_to[in_hol==0].mean():.1f}j")
print(f"  days_since_end mean        : {days_since[in_hol==0].mean():.1f}j")

# --- Splits ---
n = len(df)
split_tr = int(n * 0.70); split_cal = int(n * 0.80)
y = df[_CONSO_COL]
X_base = df.drop(columns=[_CONSO_COL]).copy()
X_base["conso_mean_52w"] = conso_mean_52w

X_tr = X_base.iloc[:split_tr]; X_te = X_base.iloc[split_cal:]
y_tr = y.iloc[:split_tr];       y_te = y.iloc[split_cal:]
sample_weight = np.exp(np.linspace(0, 2, split_tr))


def add_school(X, mask):
    X = X.copy()
    X["school_hol_A"]         = zone_a[mask]
    X["school_hol_B"]         = zone_b[mask]
    X["school_hol_C"]         = zone_c[mask]
    X["days_to_hol_start"]    = days_to[mask]
    X["days_since_hol_end"]   = days_since[mask]
    X["holiday_type"]         = hol_type[mask]
    X["holiday_progress"]     = progress[mask]
    return X

mask_tr = np.arange(split_tr)
mask_te = np.arange(split_cal, n)


def eval_model(X_train, X_test, label):
    m = XGBRegressor(objective="reg:squarederror", **MODEL_PARAMS)
    m.fit(X_train, y_tr, sample_weight=sample_weight)
    p = m.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_te, p))
    r2   = r2_score(y_te, p)
    acc  = float(np.mean(np.abs((y_te - p) / y_te) < 0.05))
    print(f"  [{label:<35}] R2={r2:.4f}  RMSE={rmse:.1f}  Acc5={acc:.4f}")
    return m


print("\n[Baseline]")
eval_model(X_tr, X_te, "baseline")

print("\n[+ toutes vacances scolaires]")
Xtr_s = add_school(X_tr, mask_tr)
Xte_s = add_school(X_te, mask_te)
m_all = eval_model(Xtr_s, Xte_s, "toutes features scolaires")

print("\n[Test features individuelles]")
for col in ["school_hol_A","school_hol_B","school_hol_C",
            "days_to_hol_start","days_since_hol_end","holiday_type","holiday_progress"]:
    Xtr2 = X_tr.copy(); Xtr2[col] = Xtr_s[col]
    Xte2 = X_te.copy(); Xte2[col] = Xte_s[col]
    eval_model(Xtr2, Xte2, f"+ {col}")

print("\n=== Top 15 importances (toutes features scolaires) ===")
imp = pd.Series(m_all.feature_importances_, index=Xtr_s.columns).nlargest(15)
for feat, val in imp.items():
    print(f"  {feat:<28} {val:.4f}")
