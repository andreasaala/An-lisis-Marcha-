"""
Extraccion de parametros de marcha del dataset ClinicalGait (Voisard et al., 2024)
a partir de los ficheros *_processed_data.txt y *_meta.json de cada trial.

VERSION 2: identica a extract_gait_features.py en el calculo de todas las
columnas originales (los mismos valores, ciclo a ciclo y trial a trial), pero
anade parametros de VARIABILIDAD ciclo a ciclo (SD y CV%) reutilizando las
mismas listas por-ciclo que ya se calculaban internamente (stride_time_L,
len_L, etc.), sin duplicar deteccion de eventos ni filtros fisiologicos. Ver
el bloque "# Variabilidad" en COLUMNS y las funciones safe_sd/safe_cv_pct.

Genera un CSV con una fila por trial (subject_session_trial) y las columnas
de metadatos + parametros biomecanicos definidos en la tabla de parametros del TFM.

Uso:
    python extract_gait_features_v2.py --data-root data --output gait_features_v2.csv

Dependencias: numpy, pandas, scipy
    pip install numpy pandas scipy
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:  # scipy < 1.6
    from scipy.integrate import cumtrapz as cumulative_trapezoid


# --------------------------------------------------------------------------
# Configuracion / constantes
# --------------------------------------------------------------------------

UTURN_MARGIN_S = 0.5            # margen (s) alrededor del giro para excluir ciclos
# Ventana +-100ms para atenuacion de impacto. Con +-50ms (valor original de la
# tabla de parametros), el pico de aceleracion de la cabeza cae truncado justo
# en el borde de la ventana en el 63.7% de los casos (validado empiricamente):
# el impacto tarda en propagarse del pie a la cabeza (mediana ~90ms de
# diferencia entre ambos picos), asi que una ventana centrada de solo 100ms
# totales es demasiado estrecha para capturar el pico de cabeza completo, lo
# que infla artificialmente la atenuacion calculada. Con +-100ms baja al
# 27.6% sin riesgo de invadir el ciclo de marcha siguiente (tiempos de paso
# tipicos ~550-600ms).
SHOCK_WINDOW_MS = 100
CUTOFF_HIGHPASS_HZ = 0.5        # paso-alto tras doble integracion (CoM)

# Limites fisiologicos plausibles: valores fuera de rango se descartan como
# fallos de integracion/deteccion, no se fuerzan a la media.
STRIDE_TIME_RANGE_S = (0.3, 3.0)
STRIDE_LEN_RANGE_M = (0.05, 2.5)
FOOT_ANGLE_RANGE_DEG = (-60.0, 60.0)
COM_AMPLITUDE_RANGE_M = (0.0, 0.30)

# Numero minimo de ciclos validos para estimar variabilidad (SD/CV) ciclo a
# ciclo. Con menos ciclos, la SD/CV estimados tienen tan poca fiabilidad
# (varianza del propio estimador es enorme con muestras pequenas) que no
# aportan informacion util; se devuelven como NaN en vez de forzar un valor
# con apariencia de precision.
MIN_CYCLES_VARIABILITY = 4

COLUMNS = [
    # Identificacion / trazabilidad
    "subject", "trial", "session", "daysSinceFirstSession",
    # Grupo / etiqueta
    "group", "pathologyKey",
    # Demografia / antropometria
    "age", "gender", "height_m", "weight_kg", "BMI", "laterality",
    # Clinico
    "clinicalDeficitSide", "evaluationScoreValue", "visualGaitAssessment", "TUG_s",
    # Contexto de adquisicion
    "sensor", "protocolDistance_m",
    # Temporal
    "cadence_stepsPerMin",
    "strideTime_L_s", "strideTime_R_s",
    "stepTime_s",
    "stancePhase_L_pct", "stancePhase_R_pct",
    "doubleSupport_pct",
    "asymmetry_strideTime_pct",
    # Espacial
    "strideLength_L_m", "strideLength_R_m",
    "gaitSpeed_ms",
    "asymmetry_strideLength_pct",
    # Cinematico
    "footStrikeAngle_L_deg", "footStrikeAngle_R_deg",
    "comVerticalAmplitude_m", "comLateralAmplitude_m",
    "shockAttenuation_L_dB", "shockAttenuation_R_dB",
    "asymmetry_shockAttenuation_dB",
    # Variabilidad
    "nCycles_L", "nCycles_R",
    "strideTime_L_cv_pct", "strideTime_R_cv_pct", "strideTime_cv_pct",
    "stepTime_cv_pct",
    "strideLength_L_cv_pct", "strideLength_R_cv_pct", "strideLength_cv_pct",
    "stancePhase_L_sd_pct", "stancePhase_R_sd_pct",
    "footStrikeAngle_L_sd_deg", "footStrikeAngle_R_sd_deg",
    "shockAttenuation_L_sd_dB", "shockAttenuation_R_sd_dB",
    "doubleSupport_sd_pct",
]


# --------------------------------------------------------------------------
# Utilidades genericas
# --------------------------------------------------------------------------

def safe_mean(values):
    return float(np.mean(values)) if len(values) else np.nan


def safe_sd(values):
    """Desviacion estandar muestral (ddof=1) de una lista de valores
    ciclo a ciclo. Devuelve NaN si hay menos de MIN_CYCLES_VARIABILITY
    ciclos validos (ver comentario junto a la constante)."""
    if len(values) < MIN_CYCLES_VARIABILITY:
        return np.nan
    return float(np.std(values, ddof=1))


def safe_cv_pct(values):
    """Coeficiente de variacion (%) = SD muestral / media * 100. Misma
    condicion de minimo de ciclos que safe_sd.

    CV frente a SD: se usa CV para magnitudes de razon estrictamente
    positivas cuya escala absoluta varia mucho entre sujetos (tiempo y
    longitud de zancada) -- normaliza la dispersion y permite comparar
    variabilidad entre sujetos con marcha rapida/lenta o zancada larga/corta.
    Se usa SD (sin normalizar) para magnitudes que ya son un porcentaje
    (fase de apoyo, doble apoyo) o que pueden acercarse a cero o cambiar de
    signo (angulo de contacto del pie, atenuacion en dB), donde un CV seria
    inestable o directamente no tendria sentido (mismo criterio que
    safe_asymmetry_pct frente a safe_diff, mas abajo)."""
    if len(values) < MIN_CYCLES_VARIABILITY:
        return np.nan
    mean = np.mean(values)
    if mean == 0:
        return np.nan
    return float(np.std(values, ddof=1) / mean * 100.0)


def safe_asymmetry_pct(a, b):
    """Asimetria relativa (%). Solo valida para magnitudes de razon
    estrictamente positivas (tiempos, longitudes): la media(a,b) nunca se
    acerca a 0, asi que el porcentaje es estable."""
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return np.nan
    denom = (a + b) / 2.0
    if denom == 0:
        return np.nan
    return abs(a - b) / denom * 100.0


def safe_diff(a, b):
    """Diferencia absoluta (misma unidad que a y b). Para magnitudes en
    escala logaritmica/dB, que pueden cruzar cero (ej. atenuacion de impacto):
    ahi un % respecto a la media es inestable (denominador cerca de 0 o con
    signos opuestos dispara el resultado), la diferencia directa no. Se usa
    el valor absoluto (no la diferencia con signo L-R) porque el indice de
    asimetria debe representar la magnitud de la asimetria, no que lado es
    mayor."""
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return np.nan
    return abs(a - b)


def cycle_overlaps_uturn(to_sample, hs_sample, uturn, freq):
    if not uturn:
        return False
    margin = UTURN_MARGIN_S * freq
    u_start, u_end = uturn
    return not (hs_sample < (u_start - margin) or to_sample > (u_end + margin))


# --------------------------------------------------------------------------
# Carga de ficheros
# --------------------------------------------------------------------------

def natural_sort_key(path: Path):
    """Ordena 'HS_2' antes que 'HS_10': trocea la ruta en fragmentos de texto
    y numero, comparando estos ultimos como enteros en vez de como texto
    (si no, un sort alfabetico deja HS_10..HS_19 pegados justo detras de
    HS_1, antes que HS_2)."""
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", str(path))]


def find_trials(data_root: Path):
    """Empareja cada *_meta.json con su *_processed_data.txt correspondiente."""
    trials = []
    missing = []
    for meta_path in sorted(data_root.rglob("*_meta.json"), key=natural_sort_key):
        processed_path = meta_path.with_name(
            meta_path.name.replace("_meta.json", "_processed_data.txt")
        )
        if processed_path.exists():
            trials.append((meta_path, processed_path))
        else:
            missing.append(str(meta_path))
    return trials, missing


def load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_processed(processed_path: Path) -> pd.DataFrame:
    # Se intenta primero separador por tabulador (formato observado en el dataset);
    # si el numero de columnas no cuadra con la cabecera, se recurre a separador
    # por espacios en blanco generico como red de seguridad.
    df = pd.read_csv(processed_path, sep="\t")
    if df.shape[1] <= 1:
        df = pd.read_csv(processed_path, sep=r"\s+", engine="python")

    # El paper dice que rellenan huecos de datos por interpolacion lineal,
    # pero un hueco pegado al final de la señal no se puede interpolar (no hay
    # muestra "despues"), y se queda como NaN en el fichero (detectado en ~51
    # trials, siempre al final, coincidiendo con la fase de espera tras el
    # final de la marcha, sin eventos de marcha ahi). Un solo NaN envenena
    # cualquier integracion acumulada (cumulative_trapezoid) desde ese punto
    # en adelante, invalidando el trial entero para nada. Se interpolan los
    # huecos internos (igual que el propio dataset) y se extienden con el
    # ultimo valor valido los huecos que tocan el principio o el final.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if df[numeric_cols].isna().any().any():
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    return df


# --------------------------------------------------------------------------
# Metadatos: normalizacion de unidades y tipos
# --------------------------------------------------------------------------

def fix_height_m(height_raw, warn):
    if height_raw is None:
        return np.nan
    h = float(height_raw)
    if h > 3.0:
        # La documentacion del dataset indica cm, pero los ficheros reales
        # contienen metros (ej. 1.79). Se detecta por magnitud y se corrige.
        warn(f"height={h} tratada como cm, convertida a m")
        return h / 100.0
    return h


def check_bmi_consistency(height_m, weight_kg, bmi_reported, warn):
    if np.isnan(height_m) or weight_kg is None or bmi_reported is None:
        return
    try:
        bmi_calc = float(weight_kg) / (height_m ** 2)
        if abs(bmi_calc - float(bmi_reported)) / float(bmi_reported) > 0.10:
            warn(f"BMI reportado ({bmi_reported}) difiere >10% del calculado ({bmi_calc:.1f})")
    except (ZeroDivisionError, TypeError):
        pass


def parse_tug(tug_raw):
    try:
        return float(tug_raw)
    except (TypeError, ValueError):
        return np.nan


def parse_protocol_distance(protocol_raw):
    if not protocol_raw:
        return np.nan
    match = re.search(r"([\d.]+)\s*m", str(protocol_raw))
    return float(match.group(1)) if match else np.nan


def extract_metadata_row(meta: dict, warn) -> dict:
    height_m = fix_height_m(meta.get("height"), warn)
    check_bmi_consistency(height_m, meta.get("weight"), meta.get("BMI"), warn)
    return {
        "subject": meta.get("subject"),
        "trial": meta.get("trial"),
        "session": meta.get("session"),
        "daysSinceFirstSession": meta.get("daysSinceFirstSession"),
        "group": meta.get("group"),
        "pathologyKey": meta.get("pathologyKey"),
        "age": meta.get("age"),
        "gender": meta.get("gender"),
        "height_m": height_m,
        "weight_kg": meta.get("weight"),
        "BMI": meta.get("BMI"),
        "laterality": meta.get("laterality"),
        "clinicalDeficitSide": meta.get("clinicalDeficitSide"),
        "evaluationScoreValue": meta.get("evaluationScoreValue"),
        "visualGaitAssessment": meta.get("visualGaitAssessment"),
        "TUG_s": parse_tug(meta.get("TUG")),
        "sensor": meta.get("sensor"),
        "protocolDistance_m": parse_protocol_distance(meta.get("protocol")),
    }


# --------------------------------------------------------------------------
# Parametros temporales (solo metadatos: leftGaitEvents / rightGaitEvents)
# --------------------------------------------------------------------------

# Politica unica de exclusion del giro: TODOS los parametros por-ciclo de esta
# seccion se calculan solo sobre el tramo recto (con el margen de
# UTURN_MARGIN_S ya usado en cycle_overlaps_uturn). Girar es un gesto motor
# distinto de caminar en linea recta (zancadas mas cortas, a veces un pivote
# casi sin avance, cadencia distinta): mezclarlo con la marcha recta sesga la
# media. Se detecto al validar la cadencia (ver mas abajo) y se aplica de
# forma uniforme para que todos los parametros describan el mismo tramo.

def straight_segments(points, uturn, freq):
    """Agrupa una lista de muestras (samples) en los tramos rectos (ida y
    vuelta por separado, sin el hueco del giro entre medias). Devuelve una
    lista de listas de puntos, una por tramo continuo fuera del giro."""
    if not points:
        return []
    if not uturn:
        return [points]
    margin = UTURN_MARGIN_S * freq
    u_start, u_end = uturn[0] - margin, uturn[1] + margin
    return [pts for pts in (
        [p for p in points if p < u_start],
        [p for p in points if p > u_end],
    ) if pts]


def straight_span_seconds(left_events, right_events, freq, uturn):
    """Tiempo real transcurrido en el tramo recto (ida + vuelta, sumadas por
    separado para no incluir el hueco del giro entre medias).

    No se calcula como suma de tiempos de zancada validos: esa via descarta
    el ciclo ENTERO (con su duracion completa) en cuanto uno de sus extremos
    roza el margen del giro, encogiendo el denominador de forma no
    controlada (se detecto asi el bug de cadencia, y el mismo mecanismo
    inflaba Doble Apoyo hasta valores >100%, imposibles)."""
    all_points = ([e[0] for e in left_events] + [e[0] for e in right_events]
                  + [e[1] for e in left_events] + [e[1] for e in right_events])
    segments = straight_segments(all_points, uturn, freq)
    return sum(max(pts) - min(pts) for pts in segments) / freq


def compute_cadence(left_events, right_events, freq, uturn):
    """Cadencia = pasos (HS de ambos pies) contados en el tramo recto,
    dividido entre el tiempo real que dura ese tramo. Validado empiricamente:
    la version por-ciclo (media de tiempos de zancada) diverge de este
    metodo (conteo real de pasos/tiempo real) en mas de un 2% en 811 de 1356
    trials, hasta un 13.5%."""
    hs_points = [e[1] for e in left_events] + [e[1] for e in right_events]
    if not hs_points:
        return np.nan
    hs_segments = straight_segments(hs_points, uturn, freq)
    n_steps = sum(len(seg) for seg in hs_segments)

    duration_min = straight_span_seconds(left_events, right_events, freq, uturn) / 60.0
    if duration_min <= 0 or n_steps == 0:
        return np.nan
    return n_steps / duration_min


def compute_stride_times(events, freq, uturn):
    """events: lista de (TO, HS). Devuelve duraciones HS_i -> HS_{i+1} en
    segundos, excluyendo ciclos que se solapen con el giro."""
    times = []
    for i in range(len(events) - 1):
        to_i, hs_i = events[i]
        hs_next = events[i + 1][1]
        if cycle_overlaps_uturn(to_i, hs_next, uturn, freq):
            continue
        dt = (hs_next - hs_i) / freq
        if STRIDE_TIME_RANGE_S[0] <= dt <= STRIDE_TIME_RANGE_S[1]:
            times.append(dt)
    return times


def compute_stance_pct(events, freq, uturn):
    stance_pct = []
    for i in range(len(events) - 1):
        to_i, hs_i = events[i]
        to_next, hs_next = events[i + 1]
        if cycle_overlaps_uturn(to_i, hs_next, uturn, freq):
            continue
        stride_time = (hs_next - hs_i) / freq
        # Mismo filtro de plausibilidad fisiologica que compute_stride_times:
        # un ciclo con stride_time fuera de rango (evento mal detectado) no
        # deberia ser valido aqui si no lo es como tiempo de zancada.
        if not (STRIDE_TIME_RANGE_S[0] <= stride_time <= STRIDE_TIME_RANGE_S[1]):
            continue
        stance = ((to_next - hs_i) / freq) / stride_time * 100.0
        if 0 < stance < 100:
            stance_pct.append(stance)
    return stance_pct


def compute_step_times(left_events, right_events, freq, uturn):
    merged = [(e[1], "L") for e in left_events] + [(e[1], "R") for e in right_events]
    merged.sort(key=lambda x: x[0])
    steps = []
    for (t0, f0), (t1, f1) in zip(merged, merged[1:]):
        if f0 != f1 and not cycle_overlaps_uturn(t0, t1, uturn, freq):
            dt = (t1 - t0) / freq
            if STRIDE_TIME_RANGE_S[0] / 2 <= dt <= STRIDE_TIME_RANGE_S[1] / 2:
                steps.append(dt)
    return steps


def compute_double_support_values(left_events, right_events, freq, uturn):
    """Lista de valores de doble apoyo (%) ciclo a ciclo, ambos pies
    combinados (misma definicion y mismos filtros que
    compute_double_support_pct, ver docstring alli). Se extrae como funcion
    independiente para poder calcular tambien la SD ciclo a ciclo
    (variabilidad), ademas de la media."""

    def stance_intervals(events):
        return [(events[i][1], events[i + 1][0]) for i in range(len(events) - 1)]

    def per_cycle_pct(own_events, other_events):
        other_stance = stance_intervals(other_events)
        values = []
        for i in range(len(own_events) - 1):
            to_i, hs_i = own_events[i]
            to_next, hs_next = own_events[i + 1]
            if cycle_overlaps_uturn(to_i, hs_next, uturn, freq):
                continue
            stride_time = (hs_next - hs_i) / freq
            if not (STRIDE_TIME_RANGE_S[0] <= stride_time <= STRIDE_TIME_RANGE_S[1]):
                continue
            own_start, own_end = hs_i, to_next
            overlap_samples = sum(
                max(0.0, min(own_end, b_end) - max(own_start, b_start))
                for b_start, b_end in other_stance
            )
            pct = (overlap_samples / freq) / stride_time * 100.0
            if 0 <= pct <= 100:
                values.append(pct)
        return values

    return per_cycle_pct(left_events, right_events) + per_cycle_pct(right_events, left_events)


def compute_double_support_pct(left_events, right_events, freq, uturn):
    """Doble apoyo ciclo a ciclo: para cada zancada valida de un pie
    [HS_k, HS_{k+1}] (misma validez que compute_stride_times: sin solape con
    el giro y duracion entre 0.3 y 3s), se mide el tiempo en que su intervalo
    de apoyo [HS_k, TO_{k+1}] se solapa con cualquier intervalo de apoyo del
    pie contrario, y se divide entre la duracion de esa misma zancada. Se
    devuelve la media de todos los ciclos validos de ambos pies.

    Version anterior (abandonada): ratio agrupado = solape total entre todos
    los intervalos de apoyo / tiempo real transcurrido en el tramo recto
    (straight_span_seconds). Ese metodo infraestimaba el resultado de forma
    sistematica (~5 puntos porcentuales por debajo de Voisard et al., 2024,
    Tabla 10, en todas las cohortes): el denominador (tiempo real del tramo)
    incluye los bordes de cada tramo recto (inicio de la marcha, zonas junto
    al giro) donde aun no hay un intervalo de apoyo completo formado para
    alguno de los dos pies, y por tanto ese tiempo no puede contabilizarse
    como doble apoyo en el numerador. Al calcular ciclo a ciclo, numerador y
    denominador se refieren siempre al mismo ciclo, eliminando ese sesgo; el
    resultado tampoco puede superar el 100% por construccion, ya que el
    intervalo de apoyo de un ciclo esta contenido en su propia zancada."""
    return safe_mean(compute_double_support_values(left_events, right_events, freq, uturn))


# --------------------------------------------------------------------------
# Parametros espaciales: longitud de zancada por doble integracion + ZUPT
# --------------------------------------------------------------------------

def find_zero_velocity_sample(gyr_xyz: np.ndarray, start: int, end: int):
    """Localiza, dentro de un intervalo de apoyo [start, end] del pie, la
    muestra de norma minima de velocidad angular (Gyr_X/Y/Z) — el instante de
    'pie plano' en el que la velocidad de traslacion es minima. Ni TO ni HS
    son validos como referencia de velocidad cero: en TO el pie ya tiene
    velocidad hacia delante (impulso de despegue) y en HS aun esta frenando
    (validado empiricamente: |Gyr| en TO y HS es varias veces mayor que en
    este minimo intermedio, que aparece de forma consistente hacia el 30% del
    apoyo)."""
    segment = gyr_xyz[start:end + 1]
    if len(segment) == 0:
        return None
    norm = np.linalg.norm(segment, axis=1)
    return start + int(np.argmin(norm))


def average_flat_foot_angle(acc_xz: np.ndarray, sample: int, half_window: int = 5) -> float:
    """Estima la inclinacion absoluta del pie (atan2(Acc_X, Acc_Z), con
    gravedad) en un instante de pie plano, promediando +-half_window
    muestras alrededor en vez de tomar una unica lectura puntual del
    acelerometro. Validado empiricamente: reduce el residuo de la rotacion
    (media del canal horizontal, que deberia ser ~0 dentro de una zancada)
    en la mayoria de los casos probados."""
    lo = max(0, sample - half_window)
    hi = min(len(acc_xz), sample + half_window + 1)
    ax_mean = acc_xz[lo:hi, 0].mean()
    az_mean = acc_xz[lo:hi, 1].mean()
    return float(np.arctan2(ax_mean, az_mean))


def integrate_zupt_sagittal(acc_x: np.ndarray, acc_z: np.ndarray, gyr_y: np.ndarray,
                             theta_start: float, theta_end: float, freq: float) -> float:
    """acc_x, acc_z: Acc_X/Acc_Z del pie CON gravedad (no FreeAcc), entre dos
    instantes de pie plano consecutivos. gyr_y: Gyr_Y (eje de rotacion
    sagital) en la misma ventana. theta_start/theta_end: inclinacion absoluta
    del pie en ambos extremos (average_flat_foot_angle).

    Metodo de Sabatini et al. (2005) restringido al plano sagital: los datos
    procesados del dataset estan en el sistema de referencia LOCAL del
    sensor, no en uno global (Voisard et al., 2024, Tabla 4); durante el
    balanceo el pie rota mucho en el plano sagital (confirmado empiricamente:
    Gyr_Y llega a +-6.5 rad/s durante el balanceo, muy por encima de Gyr_X/Z,
    y es el eje con mas energia rotacional), asi que integrar Acc_X/Acc_Z
    directamente en los ejes del sensor no representa un desplazamiento
    horizontal real.
    1. theta_start/theta_end ya estiman la inclinacion absoluta en ambos
       extremos a partir de la gravedad (atan2(Acc_X, Acc_Z); confirmado
       empiricamente que Acc_Z domina en el pie plano, ~9 m/s2, con Acc_X
       aportando una componente sustancial y consistente, ~4.5 m/s2 -- Z es
       el eje mas alineado con la vertical).
    2. Se integra **menos** Gyr_Y (el signo importa: con +Gyr_Y el angulo
       diverge de la trayectoria real y filtra casi toda la gravedad al
       canal horizontal -- validado empiricamente probando ambos signos
       sobre trials sanos: con -Gyr_Y la media del canal horizontal en una
       zancada cae de ~9 m/s2 a <1-2 m/s2) para obtener el angulo en cada
       muestra, corrigiendo la deriva de forma lineal para que el angulo
       final coincida con theta_end.
    3. Se rota (Acc_X, Acc_Z) a un eje horizontal fijo en cada instante con
       ese angulo (el eje vertical, con la gravedad restada, no hace falta
       para la longitud de zancada, que es puramente horizontal).
    4. Se integra dos veces la aceleracion horizontal con ZUPT (velocidad
       cero impuesta en ambos extremos, correccion lineal de deriva, igual
       que antes).
    Devuelve el desplazamiento horizontal neto (longitud de zancada)."""
    n = len(acc_x)
    if n < 3:
        return np.nan
    t = np.arange(n) / freq

    theta_raw = theta_start - cumulative_trapezoid(gyr_y, t, initial=0)
    drift = theta_raw[-1] - theta_end
    theta = theta_raw - drift * (t / t[-1])

    horizontal = acc_x * np.cos(theta) - acc_z * np.sin(theta)

    vel = cumulative_trapezoid(horizontal, t, initial=0)
    vel_corr = vel - (t / t[-1]) * vel[-1]
    pos = cumulative_trapezoid(vel_corr, t, initial=0)
    return float(abs(pos[-1]))


def compute_stride_lengths(events, freq, acc_xz: np.ndarray, gyr_xyz: np.ndarray, uturn):
    """acc_xz: array (N,2) con Acc_X/Acc_Z del pie (CON gravedad) para todo
    el trial. gyr_xyz: array (N,3) con Gyr_X/Y/Z del pie para todo el trial.

    Para cada ciclo i, se localiza el instante de velocidad angular minima
    dentro del apoyo anterior [HS_{i-1}, TO_i] y dentro del apoyo posterior
    [HS_i, TO_{i+1}] (find_zero_velocity_sample), se estima la inclinacion
    absoluta del pie en ambos instantes (average_flat_foot_angle) y se
    integra el desplazamiento horizontal entre ambos en el plano sagital
    (integrate_zupt_sagittal). Devuelve las longitudes validas del tramo
    recto (excluye ciclos solapados con el giro)."""
    lengths = []
    n_samples = acc_xz.shape[0]
    gyr_y = gyr_xyz[:, 1]
    for i in range(1, len(events) - 1):
        to_i, hs_i = events[i]
        prev_hs = events[i - 1][1]
        next_to = events[i + 1][0]
        if not (0 <= prev_hs < to_i < n_samples) or not (0 <= hs_i < next_to < n_samples):
            continue
        zv_start = find_zero_velocity_sample(gyr_xyz, prev_hs, to_i)
        zv_end = find_zero_velocity_sample(gyr_xyz, hs_i, next_to)
        if zv_start is None or zv_end is None or zv_end <= zv_start:
            continue
        if cycle_overlaps_uturn(zv_start, zv_end, uturn, freq):
            continue
        theta_start = average_flat_foot_angle(acc_xz, zv_start)
        theta_end = average_flat_foot_angle(acc_xz, zv_end)
        length = integrate_zupt_sagittal(
            acc_xz[zv_start:zv_end + 1, 0], acc_xz[zv_start:zv_end + 1, 1],
            gyr_y[zv_start:zv_end + 1], theta_start, theta_end, freq,
        )
        if np.isnan(length) or not (STRIDE_LEN_RANGE_M[0] <= length <= STRIDE_LEN_RANGE_M[1]):
            continue
        lengths.append(length)
    return lengths


def compute_gait_speed(len_L_straight, time_L_straight, len_R_straight, time_R_straight):
    """Velocidad = media de longitudes de zancada / media de tiempos de
    zancada (ambos pies agrupados). No se usa razon de sumas
    (sum(long)/sum(tiempo)): las listas de longitudes y de tiempos no
    contienen necesariamente los mismos ciclos (compute_stride_lengths y
    compute_stride_times aplican criterios de validez ligeramente distintos,
    con instantes de referencia distintos), y una razon de sumas con listas
    de distinto origen introducia un sesgo sistematico (~5% a la baja)."""
    lengths = len_L_straight + len_R_straight
    times = time_L_straight + time_R_straight
    if not lengths or not times:
        return np.nan
    mean_time = np.mean(times)
    if mean_time <= 0:
        return np.nan
    return np.mean(lengths) / mean_time


# --------------------------------------------------------------------------
# Parametros cinematicos
# --------------------------------------------------------------------------

def compute_foot_strike_angles(events, freq, gyr_xyz: np.ndarray, uturn):
    angles = []
    n_samples = gyr_xyz.shape[0]
    gyr_y = gyr_xyz[:, 1]
    for i in range(1, len(events)):
        to_i, hs_i = events[i]
        prev_hs = events[i - 1][1]
        # Mismo instante de pie plano que compute_stride_lengths
        # (find_zero_velocity_sample), en vez del punto medio geometrico del
        # apoyo: ambos parametros comparten ahora la misma definicion de
        # referencia "pie plano".
        mid_stance = find_zero_velocity_sample(gyr_xyz, prev_hs, to_i)
        if mid_stance is None or not (0 <= mid_stance < hs_i < n_samples):
            continue
        if cycle_overlaps_uturn(mid_stance, hs_i, uturn, freq):
            continue
        segment = gyr_y[mid_stance:hs_i + 1]
        if len(segment) < 2:
            continue
        t = np.arange(len(segment)) / freq
        angle_rad = cumulative_trapezoid(segment, t, initial=0)[-1]
        angle_deg = float(np.degrees(angle_rad))
        if FOOT_ANGLE_RANGE_DEG[0] <= angle_deg <= FOOT_ANGLE_RANGE_DEG[1]:
            angles.append(angle_deg)
    return angles


def butter_highpass_filter(signal: np.ndarray, freq: float, cutoff=CUTOFF_HIGHPASS_HZ, order=2):
    nyq = 0.5 * freq
    b, a = butter(order, cutoff / nyq, btype="high")
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return None
    return filtfilt(b, a, signal)


def compute_com_amplitude(acc_1d: np.ndarray, freq: float, cycle_windows, uturn):
    """Doble integracion + paso-alto de una senal de aceleracion (CoM),
    amplitud pico-a-pico promediada por ciclo de marcha (mas robusto que un
    unico pico-a-pico global sensible a un solo outlier).

    cycle_windows: ventanas de CICLO COMPLETO (HS_i, HS_{i+1}), no de solo
    balanceo (TO_i, HS_i). La oscilacion del CoM (vertical y lateral) tiene su
    propia periodicidad ligada al ciclo entero (dos maximos por ciclo, uno
    por cada apoyo, segun el marco teorico), no a la fase de balanceo (~40%
    del ciclo). Validado empiricamente: usar solo el balanceo subestima la
    amplitud ~23% de media frente a usar el ciclo completo, y el ciclo
    completo encaja mucho mejor con la referencia de literatura (~4-5cm de
    oscilacion vertical en sanos)."""
    t = np.arange(len(acc_1d)) / freq
    vel = cumulative_trapezoid(acc_1d, t, initial=0)
    pos = cumulative_trapezoid(vel, t, initial=0)
    pos_hp = butter_highpass_filter(pos, freq)
    if pos_hp is None:
        return []
    amplitudes = []
    for start, end in cycle_windows:
        if cycle_overlaps_uturn(start, end, uturn, freq):
            continue
        seg = pos_hp[start:end + 1]
        if len(seg) < 2:
            continue
        amp = float(seg.max() - seg.min())
        if COM_AMPLITUDE_RANGE_M[0] <= amp <= COM_AMPLITUDE_RANGE_M[1]:
            amplitudes.append(amp)
    return amplitudes


# Eje mediolateral del sensor lumbar (LB) para la oscilacion lateral del CoM.
# El paper no lo confirma (a diferencia del eje craniocaudal X). Segun el
# marco teorico, la oscilacion LATERAL tiene un unico maximo por ciclo (1x la
# frecuencia de zancada), a diferencia de la VERTICAL que tiene dos (2x). Una
# primera validacion uso por error el criterio 2x (el de la vertical) y
# concluyo Z; repetida con el criterio correcto (1x) sobre las 1356 muestras,
# el eje Y encaja mejor en el 68.2% de los trials frente al 31.8% de Z. Se
# fija en Y para todas las muestras en vez de decidir por-trial, por el mismo
# motivo que antes (evitar que un heuristico por-trial falle mas en las
# cohortes neurologicas mas severas, de mayor interes clinico).
LATERAL_AXIS = "Y"


def check_lateral_axis_plausibility(sig_y: np.ndarray, sig_z: np.ndarray, freq: float,
                                     stride_freq_hz: float, warn):
    """No decide el eje (fijo arriba en Y): solo advierte si, para este trial
    concreto, Z encajaria claramente mejor con la periodicidad de marcha
    esperada, para poder revisar esos casos concretos a mano."""
    candidates = {"Y": sig_y, "Z": sig_z}
    # La oscilacion lateral completa 1 ciclo por zancada (ver marco teorico:
    # un unico maximo por ciclo), no 2 (eso es la vertical).
    expected = 1.0 * stride_freq_hz
    scores = {}
    for axis, sig in candidates.items():
        sig = np.asarray(sig, dtype=float)
        n = len(sig)
        if n < freq:
            continue
        sig = sig - np.mean(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / freq)
        power = np.abs(np.fft.rfft(sig)) ** 2
        band = (freqs > 0.3) & (freqs < 5.0)
        if not band.any() or power[band].sum() == 0:
            continue
        dom_freq = freqs[band][np.argmax(power[band])]
        scores[axis] = abs(dom_freq - expected)
    if len(scores) == 2 and scores["Z"] < scores["Y"]:
        warn("CoM lateral: el eje Z encaja mejor que Y (fijado) en este trial; revisar a mano")


def compute_shock_attenuation(events, freq, foot_res: np.ndarray, head_res: np.ndarray, uturn):
    half_win = max(1, int(round((SHOCK_WINDOW_MS / 1000.0) * freq)))
    n = min(len(foot_res), len(head_res))
    values = []
    for _, hs_s in events:
        if cycle_overlaps_uturn(hs_s, hs_s, uturn, freq):
            continue
        lo, hi = max(0, hs_s - half_win), min(n, hs_s + half_win + 1)
        if hi <= lo:
            continue
        peak_foot = float(foot_res[lo:hi].max())
        peak_head = float(head_res[lo:hi].max())
        if peak_head <= 1e-6 or peak_foot <= 0:
            continue
        values.append(20.0 * np.log10(peak_foot / peak_head))
    return values


def resultant(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return np.sqrt(
        df[f"{prefix}_Acc_X"].to_numpy() ** 2
        + df[f"{prefix}_Acc_Y"].to_numpy() ** 2
        + df[f"{prefix}_Acc_Z"].to_numpy() ** 2
    )


# --------------------------------------------------------------------------
# Procesamiento de un trial completo
# --------------------------------------------------------------------------

def process_trial(meta_path: Path, processed_path: Path, warnings_log: list) -> dict:
    trial_id = meta_path.stem.replace("_meta", "")

    def warn(msg):
        warnings_log.append({"trial_id": trial_id, "message": msg})

    meta = load_meta(meta_path)
    row = {"trial_id": trial_id}
    row.update(extract_metadata_row(meta, warn))

    freq = float(meta.get("freq", 100.0))
    if abs(freq - 100.0) > 1e-6:
        warn(f"freq={freq} distinta del valor habitual 100Hz")

    left_events = [tuple(p) for p in (meta.get("leftGaitEvents") or [])]
    right_events = [tuple(p) for p in (meta.get("rightGaitEvents") or [])]
    uturn = meta.get("uturnBoundaries")

    if len(left_events) < 2 and len(right_events) < 2:
        warn("Menos de 2 eventos de marcha detectados en ambos pies; "
             "parametros biomecanicos no calculables")
        for col in COLUMNS:
            row.setdefault(col, np.nan)
        return row

    df = load_processed(processed_path)

    # --- Temporal (tramo recto, sin giro) ---
    stride_time_L = compute_stride_times(left_events, freq, uturn)
    stride_time_R = compute_stride_times(right_events, freq, uturn)
    step_times = compute_step_times(left_events, right_events, freq, uturn)
    stance_pct_L = compute_stance_pct(left_events, freq, uturn)
    stance_pct_R = compute_stance_pct(right_events, freq, uturn)
    double_support_values = compute_double_support_values(left_events, right_events, freq, uturn)

    row["strideTime_L_s"] = safe_mean(stride_time_L)
    row["strideTime_R_s"] = safe_mean(stride_time_R)
    row["stepTime_s"] = safe_mean(step_times)
    row["stancePhase_L_pct"] = safe_mean(stance_pct_L)
    row["stancePhase_R_pct"] = safe_mean(stance_pct_R)
    row["doubleSupport_pct"] = safe_mean(double_support_values)
    row["cadence_stepsPerMin"] = compute_cadence(left_events, right_events, freq, uturn)
    row["asymmetry_strideTime_pct"] = safe_asymmetry_pct(row["strideTime_L_s"], row["strideTime_R_s"])

    # --- Variabilidad temporal (ciclo a ciclo, mismas listas de arriba) ---
    row["nCycles_L"] = len(stride_time_L)
    row["nCycles_R"] = len(stride_time_R)
    row["strideTime_L_cv_pct"] = safe_cv_pct(stride_time_L)
    row["strideTime_R_cv_pct"] = safe_cv_pct(stride_time_R)
    row["strideTime_cv_pct"] = safe_cv_pct(stride_time_L + stride_time_R)
    row["stepTime_cv_pct"] = safe_cv_pct(step_times)
    row["stancePhase_L_sd_pct"] = safe_sd(stance_pct_L)
    row["stancePhase_R_sd_pct"] = safe_sd(stance_pct_R)
    row["doubleSupport_sd_pct"] = safe_sd(double_support_values)

    # --- Espacial (tramo recto, sin giro) ---
    try:
        lf_acc_xz = df[["LF_Acc_X", "LF_Acc_Z"]].to_numpy()
        rf_acc_xz = df[["RF_Acc_X", "RF_Acc_Z"]].to_numpy()
        lf_gyr = df[["LF_Gyr_X", "LF_Gyr_Y", "LF_Gyr_Z"]].to_numpy()
        rf_gyr = df[["RF_Gyr_X", "RF_Gyr_Y", "RF_Gyr_Z"]].to_numpy()
        len_L = compute_stride_lengths(left_events, freq, lf_acc_xz, lf_gyr, uturn)
        len_R = compute_stride_lengths(right_events, freq, rf_acc_xz, rf_gyr, uturn)
        row["strideLength_L_m"] = safe_mean(len_L)
        row["strideLength_R_m"] = safe_mean(len_R)
        row["asymmetry_strideLength_pct"] = safe_asymmetry_pct(
            row["strideLength_L_m"], row["strideLength_R_m"]
        )
        row["gaitSpeed_ms"] = compute_gait_speed(len_L, stride_time_L, len_R, stride_time_R)
        row["strideLength_L_cv_pct"] = safe_cv_pct(len_L)
        row["strideLength_R_cv_pct"] = safe_cv_pct(len_R)
        row["strideLength_cv_pct"] = safe_cv_pct(len_L + len_R)
    except KeyError as exc:
        warn(f"Columnas Acc/Gyr de pie no encontradas: {exc}")
        row["strideLength_L_m"] = row["strideLength_R_m"] = np.nan
        row["asymmetry_strideLength_pct"] = row["gaitSpeed_ms"] = np.nan
        row["strideLength_L_cv_pct"] = row["strideLength_R_cv_pct"] = np.nan
        row["strideLength_cv_pct"] = np.nan

    # --- Cinematico: angulo de contacto del pie ---
    try:
        lf_gyr_fsa = df[["LF_Gyr_X", "LF_Gyr_Y", "LF_Gyr_Z"]].to_numpy()
        rf_gyr_fsa = df[["RF_Gyr_X", "RF_Gyr_Y", "RF_Gyr_Z"]].to_numpy()
        angles_L = compute_foot_strike_angles(left_events, freq, lf_gyr_fsa, uturn)
        angles_R = compute_foot_strike_angles(right_events, freq, rf_gyr_fsa, uturn)
        row["footStrikeAngle_L_deg"] = safe_mean(angles_L)
        row["footStrikeAngle_R_deg"] = safe_mean(angles_R)
        row["footStrikeAngle_L_sd_deg"] = safe_sd(angles_L)
        row["footStrikeAngle_R_sd_deg"] = safe_sd(angles_R)
    except KeyError as exc:
        warn(f"Columnas Gyr_Y de pie no encontradas: {exc}")
        row["footStrikeAngle_L_deg"] = row["footStrikeAngle_R_deg"] = np.nan
        row["footStrikeAngle_L_sd_deg"] = row["footStrikeAngle_R_sd_deg"] = np.nan

    # --- Cinematico: desplazamiento del CoM (sensor lumbar LB) ---
    try:
        ref_events = left_events if len(left_events) >= len(right_events) else right_events
        cycle_windows = [(ref_events[i][1], ref_events[i + 1][1]) for i in range(len(ref_events) - 1)]
        stride_pool = stride_time_L + stride_time_R
        stride_freq_hz = 1.0 / np.mean(stride_pool) if stride_pool else 1.0

        row["comVerticalAmplitude_m"] = safe_mean(
            compute_com_amplitude(df["LB_FreeAcc_X"].to_numpy(), freq, cycle_windows, uturn)
        )
        check_lateral_axis_plausibility(
            df["LB_FreeAcc_Y"].to_numpy(), df["LB_FreeAcc_Z"].to_numpy(),
            freq, stride_freq_hz, warn,
        )
        row["comLateralAmplitude_m"] = safe_mean(
            compute_com_amplitude(df[f"LB_FreeAcc_{LATERAL_AXIS}"].to_numpy(), freq, cycle_windows, uturn)
        )
    except KeyError as exc:
        warn(f"Columnas FreeAcc lumbares no encontradas: {exc}")
        row["comVerticalAmplitude_m"] = row["comLateralAmplitude_m"] = np.nan

    # --- Cinematico: atenuacion del impacto ---
    try:
        head_res = resultant(df, "HE")
        lf_res = resultant(df, "LF")
        rf_res = resultant(df, "RF")
        shock_L = compute_shock_attenuation(left_events, freq, lf_res, head_res, uturn)
        shock_R = compute_shock_attenuation(right_events, freq, rf_res, head_res, uturn)
        row["shockAttenuation_L_dB"] = safe_mean(shock_L)
        row["shockAttenuation_R_dB"] = safe_mean(shock_R)
        row["asymmetry_shockAttenuation_dB"] = safe_diff(
            row["shockAttenuation_L_dB"], row["shockAttenuation_R_dB"]
        )
        row["shockAttenuation_L_sd_dB"] = safe_sd(shock_L)
        row["shockAttenuation_R_sd_dB"] = safe_sd(shock_R)
    except KeyError as exc:
        warn(f"Columnas Acc no encontradas: {exc}")
        row["shockAttenuation_L_dB"] = row["shockAttenuation_R_dB"] = np.nan
        row["asymmetry_shockAttenuation_dB"] = np.nan
        row["shockAttenuation_L_sd_dB"] = row["shockAttenuation_R_sd_dB"] = np.nan

    for col in COLUMNS:
        row.setdefault(col, np.nan)
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrae parametros de marcha del dataset ClinicalGait a un CSV."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                         help="Carpeta raiz con los subdirectorios healthy/ortho/neuro")
    parser.add_argument("--output", type=Path, default=Path("gait_features_v2.csv"))
    parser.add_argument("--log", type=Path, default=Path("extraction_warnings_v2.csv"))
    parser.add_argument("--limit", type=int, default=None,
                         help="Procesar solo los N primeros trials (pruebas rapidas)")
    args = parser.parse_args()

    trials, missing_processed = find_trials(args.data_root)
    if args.limit:
        trials = trials[: args.limit]

    print(f"Trials encontrados: {len(trials)} (sin processed_data: {len(missing_processed)})")

    rows = []
    warnings_log = [{"trial_id": "GLOBAL", "message": f"processed_data faltante: {p}"}
                     for p in missing_processed]

    for i, (meta_path, processed_path) in enumerate(trials, 1):
        try:
            row = process_trial(meta_path, processed_path, warnings_log)
        except Exception as exc:  # nunca se debe perder un trial silenciosamente
            warnings_log.append({
                "trial_id": meta_path.stem.replace("_meta", ""),
                "message": f"FALLO TOTAL: {exc}\n{traceback.format_exc(limit=3)}",
            })
            row = {"trial_id": meta_path.stem.replace("_meta", "")}
            row.update(extract_metadata_row(load_meta(meta_path), lambda m: None))
            for col in COLUMNS:
                row.setdefault(col, np.nan)
        rows.append(row)

        if i % 50 == 0 or i == len(trials):
            print(f"  procesados {i}/{len(trials)}")

    out_columns = ["trial_id"] + COLUMNS
    df_out = pd.DataFrame(rows)[out_columns]
    df_out.to_csv(args.output, index=False)

    df_warn = pd.DataFrame(warnings_log)
    df_warn.to_csv(args.log, index=False)

    print(f"\nCSV generado: {args.output}  ({len(df_out)} filas x {len(out_columns)} columnas)")
    print(f"Log de incidencias: {args.log}  ({len(df_warn)} entradas)")


if __name__ == "__main__":
    main()
