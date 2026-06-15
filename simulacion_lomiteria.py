from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Simulación Lomitería – TP5", layout="wide")

INF = float("inf")

# ── helpers ──────────────────────────────────────────────────────────────────

def fmt(seg: float) -> str:
    s = int(round(seg))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def normal_pos(media: float, desv: float, rng: random.Random):
    while True:
        r1, r2 = rng.random(), rng.random()
        z = math.sqrt(-2 * math.log(r1)) * math.cos(2 * math.pi * r2)
        v = media + desv * z
        if v > 0:
            return r1, r2, v


def uniforme(a: float, b: float, rng: random.Random):
    r = rng.random()
    return r, a + r * (b - a)


def unif_disc(a: int, b: int, rng: random.Random):
    r = rng.random()
    return r, min(a + int(r * (b - a + 1)), b)


def permanencia_params(salon: str, reloj: float):
    """Devuelve (media_min, desvio_min) en segundos según franja horaria."""
    h = reloj / 3600
    tabla = {
        "rojo":  [(11,12,20,15),(12,13,30,15),(13,14,35,15),(14,15,20,15)],
        "azul":  [(11,12,30,15),(12,13,40,15),(13,14,45,10),(14,15,35,15)],
    }
    for (h0, h1, med, dev) in tabla[salon]:
        if h0 <= h < h1:
            return med * 60, dev * 60
    return 20 * 60, 15 * 60  # fallback


def rk_local(a_val: int, h: float = 0.01):
    def f(t, l): return 6 + 3 * a_val
    t, l = 0.0, float(a_val)
    rows = []
    while l <= 10:
        k1 = f(t, l)
        k2 = f(t + h/2, l + h*k1/2)
        k3 = f(t + h/2, l + h*k2/2)
        k4 = f(t + h, l + h*k3)
        l2 = l + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t2 = t + h
        rows.append({"A": a_val, "t": round(t,4), "L": round(l,4),
                     "k1": round(k1,4), "k2": round(k2,4),
                     "k3": round(k3,4), "k4": round(k4,4),
                     "t_sig": round(t2,4), "L_sig": round(l2,4)})
        t, l = t2, l2
    return t * 600, rows   # segundos

# ── simulación ────────────────────────────────────────────────────────────────

RENAME = {
    "r1_llg": "RND 1 llegada",
    "r2_llg": "RND 2 llegada",
    "tel": "Tiempo entre llegadas",
    "r_tipo": "RND tipo",
    "r_caja": "RND atencion",
    "ta_caja": "Tiempo atencion",
    "r_A": "RND CL",
    "A": "Valor de A",
    "tp_prep": "Tiempo de Preparacion CL",
    "r_llevar": "RND Para Llevar",
    "tp_llevar": "Tiempo Preparacion Llevar",
    "r_salon": "RND Salon",
    "r_sal1": "RND1 Permanencia",
    "r_sal2": "RND2 Permanencia",
    "tp_sal": "Tiempo Permanencia Salon",
    "cola_caja": "Cola Caja",
    "cola_most": "Cola Mostrador",
    "r_ocup": "Salon Rojo Ocupados",
    "a_ocup": "Salon Azul Ocupados",
    "vivos": "Clientes Vivos",
}


def simular(p: dict) -> dict:
    rng = random.Random(p["semilla"])
    reloj = p["hora_inicio"]
    fin    = p["hora_fin"]
    it     = 0
    nxt_id = 1

    # Caja
    caja_est = "libre"; caja_cli = None; caja_ini = None; caja_ac = 0.0
    cola_caja: deque = deque(); max_cc = 0

    # Mostrador (3 empleados)
    mostr = [{"est":"libre","cli":None,"ini":None,"fin":INF,"ac":0.0} for _ in range(3)]
    cola_most: deque = deque(); max_cm = 0

    # Salones
    salon_r: set = set(); salon_a: set = set()
    esp_r: deque = deque(); esp_a: deque = deque()
    salida_sal: dict = {}
    salon_r_lleno_ini: Optional[float] = None
    salon_a_lleno_ini: Optional[float] = None
    ac_r_lleno = 0.0; ac_a_lleno = 0.0

    clientes: Dict[int, dict] = {}
    vector: List[dict] = []
    rk_rows: List[dict] = []
    ctrl_most: List[dict] = []
    ctrl_sal: List[dict] = []

    # Acumuladores métricas
    ac_perm = 0.0; n_perm = 0
    ac_cc   = 0.0; n_cc   = 0
    ac_cm   = 0.0; n_cm   = 0
    n_llevar= 0;   n_local= 0; n_rojo= 0; n_azul= 0

    r1,r2,tel = normal_pos(p["media_llg"], p["desv_llg"], rng)
    prox_llg   = reloj + tel
    prox_ctrl_m = p["hora_inicio"] + p["ctrl_most"]
    prox_ctrl_s = p["hora_inicio"] + p["ctrl_sal"]

    last: dict = {}

    def libre_most():
        for i, e in enumerate(mostr):
            if e["est"] == "libre": return i
        return None

    def ini_caja(cid, ahora):
        nonlocal caja_est, caja_cli, caja_ini, ac_cc, n_cc
        c = clientes[cid]
        c["est"] = "en_caja"; c["ini_caja"] = ahora
        c["t_cc"] = ahora - c["llg"]
        ac_cc += c["t_cc"]; n_cc += 1
        r, ta = uniforme(p["caja_a"], p["caja_b"], rng)
        c["fin_caja"] = ahora + ta
        caja_est = "ocupado"; caja_cli = cid; caja_ini = ahora
        last.update({"r_caja": r, "ta_caja": ta})

    def ini_prep(cid, ei, ahora):
        nonlocal ac_cm, n_cm
        c = clientes[cid]
        c["est"] = "prep"; c["ini_prep"] = ahora
        c["t_cm"] = ahora - c["llg_most"]
        ac_cm += c["t_cm"]; n_cm += 1
        if c["tipo"] == "llevar":
            r, tp = uniforme(p["llevar_a"], p["llevar_b"], rng)
            fin_pl = ahora + tp
            last.update({"r_llevar": r, "tp_llevar": tp, "fin_prep_llevar": round(fin_pl, 2)})
        else:
            ra, av = unif_disc(2, 5, rng)
            tp, rk = rk_local(av, p["h_rk"])
            rk_rows.extend(rk)
            last.update({"r_A": ra, "A": av, "tp": tp, "fin_prep_local": round(ahora + tp, 2)})
        c["fin_prep"] = ahora + tp
        mostr[ei].update({"est":"ocupado","cli":cid,"ini":ahora,"fin":c["fin_prep"]})

    def intentar_salon(cid, ahora):
        c = clientes[cid]
        if c["salon"] == "rojo":
            if len(salon_r) < p["cap_r"]: ingresar_salon(cid, ahora)
            else: c["est"] = "esp_r"; esp_r.append(cid)
        else:
            if len(salon_a) < p["cap_a"]: ingresar_salon(cid, ahora)
            else: c["est"] = "esp_a"; esp_a.append(cid)

    def ingresar_salon(cid, ahora):
        nonlocal salon_r_lleno_ini, salon_a_lleno_ini, ac_r_lleno, ac_a_lleno
        c = clientes[cid]
        c["est"] = "en_salon"; c["ini_salon"] = ahora
        med, dev = permanencia_params(c["salon"], ahora)
        r1s, r2s, tp = normal_pos(med, dev, rng)
        c["fin_salon"] = ahora + tp
        salida_sal[cid] = c["fin_salon"]
        last.update({"r_sal1": r1s, "r_sal2": r2s, "tp_sal": tp})
        if c["salon"] == "rojo":
            salon_r.add(cid)
            if len(salon_r) >= p["cap_r"] and salon_r_lleno_ini is None:
                salon_r_lleno_ini = ahora
        else:
            salon_a.add(cid)
            if len(salon_a) >= p["cap_a"] and salon_a_lleno_ini is None:
                salon_a_lleno_ini = ahora

    def cerrar(cid, ahora):
        nonlocal ac_perm, n_perm, salon_r_lleno_ini, salon_a_lleno_ini, ac_r_lleno, ac_a_lleno
        c = clientes.pop(cid)
        c["est"] = "fue"; c["salida"] = ahora
        c["perm"] = ahora - c["llg"]
        ac_perm += c["perm"]; n_perm += 1
        # track salón lleno
        if c.get("salon") == "rojo" and salon_r_lleno_ini is not None and len(salon_r) < p["cap_r"]:
            ac_r_lleno += ahora - salon_r_lleno_ini
            salon_r_lleno_ini = None
        if c.get("salon") == "azul" and salon_a_lleno_ini is not None and len(salon_a) < p["cap_a"]:
            ac_a_lleno += ahora - salon_a_lleno_ini
            salon_a_lleno_ini = None

    def snap(evento):
        row = {
            "iteracion": it,
            "evento": evento,
            "reloj_seg": round(reloj, 2),
            "hora": fmt(reloj),
            # llegada
            "r1_llg": last.get("r1_llg"), "r2_llg": last.get("r2_llg"),
            "tel": last.get("tel"),
            "prox_llg": round(prox_llg,2) if prox_llg < INF else None,
            # tipo pedido
            "r_tipo": last.get("r_tipo"), "tipo_pedido": last.get("tipo_pedido"),
            # caja
            "r_caja": last.get("r_caja"), "ta_caja": last.get("ta_caja"),
            "fin_caja": clientes[caja_cli]["fin_caja"] if caja_cli in clientes else None,
            # rk / llevar
            "r_A": last.get("r_A"), "A": last.get("A"),
            "r_llevar": last.get("r_llevar"),
            "tp_prep": last.get("tp"),
            "fin_prep_local":  last.get("fin_prep_local"),
            "tp_llevar":       last.get("tp_llevar"),
            "fin_prep_llevar": last.get("fin_prep_llevar"),
            "salon_elegido":   last.get("salon_elegido"),
            # salon rnd
            "r_salon": last.get("r_salon"),
            "r_sal1": last.get("r_sal1"), "r_sal2": last.get("r_sal2"),
            "tp_sal": last.get("tp_sal"),
            # caja estado
            "caja_est": caja_est, "caja_cli": caja_cli,
            "cola_caja": len(cola_caja), "max_cola_caja": max_cc,
            "caja_ac": round(caja_ac,2),
            # mostrador
            **{f"m{i+1}_est":   mostr[i]["est"]               for i in range(3)},
            **{f"m{i+1}_cli":   mostr[i]["cli"]               for i in range(3)},
            **{f"m{i+1}_fin":   None if mostr[i]["fin"]==INF else round(mostr[i]["fin"],2) for i in range(3)},
            **{f"m{i+1}_ac":    round(mostr[i]["ac"],2)       for i in range(3)},
            "cola_most": len(cola_most), "max_cola_most": max_cm,
            # salones
            "r_ocup": len(salon_r), "a_ocup": len(salon_a),
            "r_esp": len(esp_r),    "a_esp": len(esp_a),
            "r_ac_lleno": round(ac_r_lleno,2), "a_ac_lleno": round(ac_a_lleno,2),
            # métricas
            "ac_perm": round(ac_perm,2),  "n_perm":  n_perm,
            "ac_cc":   round(ac_cc,2),    "n_cc":    n_cc,
            "ac_cm":   round(ac_cm,2),    "n_cm":    n_cm,
            "n_llevar": n_llevar, "n_local": n_local,
            "n_rojo": n_rojo,     "n_azul":  n_azul,
            "vivos": len(clientes),
        }
        # clientes vivos (hasta 3 visibles)
        vivos = list(clientes.values())[:3]
        for k, cv in enumerate(vivos, 1):
            row[f"cli{k}_id"]    = cv["id"]
            row[f"cli{k}_est"]   = cv["est"]
            row[f"cli{k}_llg"]   = fmt(cv["llg"])
            row[f"cli{k}_t_cc"]  = round(cv.get("t_cc", 0), 2)
            row[f"cli{k}_perm"]  = round(cv.get("perm", 0), 2)
        vector.append(row)

    # fila inicial
    snap("inicializacion")

    while it < p["max_it"]:
        last = {}
        fc = clientes[caja_cli]["fin_caja"] if caja_cli in clientes else INF
        fp = min(m["fin"] for m in mostr)
        fs = min(salida_sal.values()) if salida_sal else INF

        times = {
            "llegada":        prox_llg,
            "fin_caja":       fc,
            "fin_prep":       fp,
            "salida_salon":   fs,
            "ctrl_most":      prox_ctrl_m,
            "ctrl_sal":       prox_ctrl_s,
            "fin_sim":        fin,
        }
        ev = min(times, key=times.get)
        reloj = times[ev]
        it += 1

        if reloj >= fin:
            reloj = fin; ev = "fin_sim"

        if ev == "llegada":
            c = {"id": nxt_id, "llg": reloj, "est": "cola_caja",
                 "tipo": None, "salon": None, "ini_caja": None,
                 "fin_caja": None, "llg_most": None,
                 "t_cc": 0.0, "t_cm": 0.0, "perm": 0.0}
            clientes[nxt_id] = c
            if caja_est == "libre":
                ini_caja(nxt_id, reloj)
            else:
                cola_caja.append(nxt_id); max_cc = max(max_cc, len(cola_caja))
            nxt_id += 1
            r1,r2,tel = normal_pos(p["media_llg"], p["desv_llg"], rng)
            prox_llg = reloj + tel
            last.update({"r1_llg":r1,"r2_llg":r2,"tel":tel})

        elif ev == "fin_caja":
            nonlocal_cid = caja_cli
            c = clientes[nonlocal_cid]
            c["est"] = "cola_most"; c["llg_most"] = reloj
            rt = rng.random()
            if rt < p["prob_llevar"]:
                c["tipo"] = "llevar"; n_llevar += 1
                last.update({"r_tipo":rt,"tipo_pedido":"llevar"})
            else:
                c["tipo"] = "local"; n_local += 1
                rs = rng.random()
                c["salon"] = "rojo" if rs < p["prob_rojo"] else "azul"
                if c["salon"] == "rojo": n_rojo += 1
                else: n_azul += 1
                last.update({"r_tipo":rt,"tipo_pedido":"local","r_salon":rs,"salon_elegido":c["salon"]})
            # liberar caja
            if caja_ini is not None:
                caja_ac += reloj - caja_ini
            if cola_caja:
                sig = cola_caja.popleft(); ini_caja(sig, reloj)
            else:
                caja_est="libre"; caja_cli=None; caja_ini=None
            # mostrador
            ei = libre_most()
            if ei is not None:
                ini_prep(nonlocal_cid, ei, reloj)
            else:
                cola_most.append(nonlocal_cid); max_cm = max(max_cm, len(cola_most))

        elif ev == "fin_prep":
            fins = [m["fin"] for m in mostr]
            ei = fins.index(min(fins))
            cid = mostr[ei]["cli"]
            if mostr[ei]["ini"] is not None:
                mostr[ei]["ac"] += reloj - mostr[ei]["ini"]
            mostr[ei].update({"est":"libre","cli":None,"ini":None,"fin":INF})
            c = clientes[cid]
            if c["tipo"] == "llevar":
                cerrar(cid, reloj)
            else:
                intentar_salon(cid, reloj)
            if cola_most:
                sig = cola_most.popleft(); ini_prep(sig, ei, reloj)

        elif ev == "salida_salon":
            cid = min(salida_sal, key=salida_sal.get)
            salida_sal.pop(cid)
            c = clientes.get(cid)
            if c:
                s = c["salon"]
                (salon_r if s=="rojo" else salon_a).discard(cid)
                cerrar(cid, reloj)
                q = esp_r if s=="rojo" else esp_a
                if q: ingresar_salon(q.popleft(), reloj)

        elif ev == "ctrl_most":
            ctrl_most.append({"reloj":round(reloj,2),"hora":fmt(reloj),"cola_most":len(cola_most)})
            prox_ctrl_m += p["ctrl_most"]

        elif ev == "ctrl_sal":
            ctrl_sal.append({"reloj":round(reloj,2),"hora":fmt(reloj),
                              "salon_rojo":len(salon_r),"salon_azul":len(salon_a)})
            prox_ctrl_s += p["ctrl_sal"]

        snap(ev)
        if ev == "fin_sim": break

    dur = fin - p["hora_inicio"]
    ac_m = sum(m["ac"] for m in mostr)
    metricas = {
        "Prom. permanencia negocio (seg)":  ac_perm/n_perm if n_perm else 0,
        "Prom. tiempo cola caja (seg)":     ac_cc/n_cc     if n_cc   else 0,
        "Prom. tiempo cola mostrador (seg)":ac_cm/n_cm     if n_cm   else 0,
        "% Ocupación caja":                 caja_ac/dur*100,
        "% Ocupación empleados mostrador":  ac_m/(3*dur)*100,
        "Máx. cola caja":                   max_cc,
        "Máx. cola mostrador":              max_cm,
        "AC tiempo salón rojo lleno (seg)": ac_r_lleno,
        "AC tiempo salón azul lleno (seg)": ac_a_lleno,
        "Clientes finalizados":             n_perm,
        "Para llevar":                      n_llevar,
        "En local":                         n_local,
        "Salón rojo":                       n_rojo,
        "Salón azul":                       n_azul,
    }
    df_vector = pd.DataFrame(vector).rename(columns=RENAME)
    return {
        "vector": df_vector,
        "ctrl_most": pd.DataFrame(ctrl_most) if ctrl_most else pd.DataFrame(),
        "ctrl_sal":  pd.DataFrame(ctrl_sal)  if ctrl_sal  else pd.DataFrame(),
        "rk":        pd.DataFrame(rk_rows)   if rk_rows   else pd.DataFrame(),
        "metricas":  metricas,
    }

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🥩 Simulación Lomitería – TP5 Grupo 22")

with st.sidebar:
    st.header("⚙️ Parámetros")

    semilla   = st.number_input("Semilla aleatoria", value=22, step=1)

    st.subheader("Tiempo simulación")
    h_ini = st.number_input("Hora inicio", 0, 23, 11)
    h_fin = st.number_input("Hora fin",    0, 24, 15)

    st.subheader("Llegada de clientes (Normal)")
    media_llg = st.number_input("Media (seg)", value=60.0, min_value=1.0)
    desv_llg  = st.number_input("Desvío (seg)", value=30.0, min_value=0.1)

    st.subheader("Atención en caja (Uniforme)")
    caja_a = st.number_input("Mín caja (seg)", value=15.0, min_value=0.0)
    caja_b = st.number_input("Máx caja (seg)", value=45.0, min_value=0.1)
    if caja_b <= caja_a:
        st.warning("Máx debe ser > Mín")

    st.subheader("Tipo de pedido")
    prob_llevar = st.slider("% Para llevar", 0, 100, 25) / 100

    st.subheader("Preparación para llevar (Uniforme)")
    llevar_a = st.number_input("Mín llevar (seg)", value=100.0, min_value=0.0)
    llevar_b = st.number_input("Máx llevar (seg)", value=140.0, min_value=0.1)

    st.subheader("Runge-Kutta (consumo local)")
    h_rk = st.number_input("Paso h", value=0.01, min_value=0.0001, format="%.4f")

    st.subheader("Salones")
    prob_rojo = st.slider("% Salón Rojo", 0, 100, 30) / 100
    cap_r     = st.number_input("Capacidad salón Rojo", value=30, min_value=1)
    cap_a     = st.number_input("Capacidad salón Azul", value=40, min_value=1)

    st.subheader("Controles periódicos")
    ctrl_most_min = st.number_input("Cada N min → cola mostrador", value=15, min_value=1)
    ctrl_sal_min  = st.number_input("Cada N min → salones", value=30, min_value=1)

    st.subheader("Iteraciones")
    max_it = st.number_input("Máx iteraciones", value=100000, min_value=1, step=1000)

    st.subheader("Filtro vector estado")
    j_it = st.number_input("Desde iteración j", value=0, min_value=0, step=1)
    i_it = st.number_input("Mostrar i iteraciones", value=50, min_value=1, step=10)

    correr = st.button("▶ Simular", type="primary", use_container_width=True)

# Validaciones básicas
errores = []
if h_fin <= h_ini: errores.append("Hora fin debe ser mayor que hora inicio.")
if caja_b <= caja_a: errores.append("Máx caja debe ser > Mín caja.")
if llevar_b <= llevar_a: errores.append("Máx llevar debe ser > Mín llevar.")

if errores:
    for e in errores:
        st.error(e)
    st.stop()

if correr:
    params = dict(
        semilla=int(semilla),
        hora_inicio=int(h_ini)*3600, hora_fin=int(h_fin)*3600,
        media_llg=media_llg, desv_llg=desv_llg,
        caja_a=caja_a, caja_b=caja_b,
        prob_llevar=prob_llevar,
        llevar_a=llevar_a, llevar_b=llevar_b,
        h_rk=h_rk,
        prob_rojo=prob_rojo, cap_r=int(cap_r), cap_a=int(cap_a),
        ctrl_most=ctrl_most_min*60, ctrl_sal=ctrl_sal_min*60,
        max_it=int(max_it),
    )
    with st.spinner("Simulando..."):
        res = simular(params)
    st.session_state["res"] = res
    st.session_state["j"]   = int(j_it)
    st.session_state["i"]   = int(i_it)
    st.success(f"Simulación completada – {len(res['vector'])} iteraciones")

if "res" not in st.session_state:
    st.info("Configurá los parámetros en el panel izquierdo y presioná **Simular**.")
    st.stop()

res: dict = st.session_state["res"]
j = st.session_state["j"]
i = st.session_state["i"]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Vector de Estado",
    "📊 Métricas",
    "🕐 Controles Periódicos",
    "🔢 Runge-Kutta",
    "📈 Gráficos",
])

# ── TAB 1: Vector de Estado ───────────────────────────────────────────────────
with tab1:
    df = res["vector"]
    total = len(df)

    st.markdown(f"**Total de filas:** {total}  |  Mostrando filas **{j}** a **{min(j+i, total)-1}**")

    # filtro j..j+i + última fila
    subset = df.iloc[j:j+i]
    ultima = df.iloc[[-1]]

    st.subheader("Iteraciones seleccionadas")
    st.dataframe(subset.fillna("").reset_index(drop=True), use_container_width=True, height=400, hide_index=True)

    st.subheader("Última fila (fin de simulación)")
    # en última fila ocultar columnas temporales de RNDs
    cols_temp = ["RND 1 llegada","RND 2 llegada","Tiempo entre llegadas",
                 "RND tipo","RND atencion","Tiempo atencion",
                 "RND CL","Valor de A","RND Para Llevar","Tiempo Preparacion Llevar",
                 "Tiempo de Preparacion CL","RND Salon","RND1 Permanencia","RND2 Permanencia","Tiempo Permanencia Salon"]
    st.dataframe(ultima.drop(columns=[c for c in cols_temp if c in ultima.columns]).fillna("").reset_index(drop=True),
                 use_container_width=True, hide_index=True)

# ── TAB 2: Métricas ───────────────────────────────────────────────────────────
with tab2:
    m = res["metricas"]
    cols = st.columns(3)
    items = list(m.items())
    for k, (nombre, val) in enumerate(items):
        with cols[k % 3]:
            if isinstance(val, float):
                st.metric(nombre, f"{val:.2f}")
            else:
                st.metric(nombre, val)

# ── TAB 3: Controles Periódicos ───────────────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cola mostrador (cada 15 min)")
        if not res["ctrl_most"].empty:
            st.dataframe(res["ctrl_most"], use_container_width=True)
        else:
            st.info("Sin datos")
    with c2:
        st.subheader("Salones (cada 30 min)")
        if not res["ctrl_sal"].empty:
            st.dataframe(res["ctrl_sal"], use_container_width=True)
        else:
            st.info("Sin datos")

# ── TAB 4: Runge-Kutta ────────────────────────────────────────────────────────
with tab4:
    st.subheader("Tablas Runge-Kutta (pedidos en local)")
    if not res["rk"].empty:
        a_vals = sorted(res["rk"]["A"].unique())
        sel = st.selectbox("Ver tabla para A =", a_vals)
        st.dataframe(res["rk"][res["rk"]["A"]==sel], use_container_width=True, height=350)
        st.caption(f"Total de pasos RK calculados: {len(res['rk'])}")
    else:
        st.info("No se calcularon tablas RK en esta simulación")

# ── TAB 5: Gráficos ───────────────────────────────────────────────────────────
with tab5:
    df = res["vector"]

    st.subheader("Cola caja a lo largo del tiempo")
    st.line_chart(df.set_index("reloj_seg")["Cola Caja"])

    st.subheader("Cola mostrador a lo largo del tiempo")
    st.line_chart(df.set_index("reloj_seg")["Cola Mostrador"])

    st.subheader("Ocupación salones")
    sal_df = df[["reloj_seg","Salon Rojo Ocupados","Salon Azul Ocupados"]]
    st.line_chart(sal_df.set_index("reloj_seg"))

    st.subheader("Clientes vivos en el sistema")
    st.area_chart(df.set_index("reloj_seg")["Clientes Vivos"])