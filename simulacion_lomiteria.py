from __future__ import annotations

# =============================================================================
# Simulación Lomitería – TP5 Grupo 22
# =============================================================================
# Este programa implementa una simulación de eventos discretos para una lomitería.
# El flujo general del cliente es:
# 1) Llega al negocio.
# 2) Hace cola y paga en caja.
# 3) Pasa al mostrador para esperar/preparar su pedido.
# 4) Si es para llevar, se va cuando termina la preparación.
# 5) Si consume en el local, elige salón rojo o azul, permanece un tiempo y luego se va.
#
# Además, el aplicativo muestra:
# - Vector de estado.
# - Métricas finales.
# - Controles periódicos cada cierto tiempo.
# - Tablas de Runge-Kutta para pedidos consumidos en local.
# - Gráficos simples de evolución del sistema.

# Librerías matemáticas y aleatorias.
import math
import random
import re
from collections import deque

# Estas importaciones quedaron disponibles por si se quisiera modelar objetos con dataclasses.
# En esta versión del código no se usan directamente.
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# pandas se usa para construir tablas: vector de estado, controles y Runge-Kutta.
import pandas as pd

# Streamlit se usa para construir la interfaz web interactiva.
import streamlit as st

# Configuración general de la página Streamlit.
st.set_page_config(page_title="Simulación Lomitería – TP5", layout="wide")

# Valor infinito usado para indicar que un evento no está programado.
# Por ejemplo, si la caja está libre, no existe un próximo fin de atención de caja.
INF = float("inf")

# ── helpers ──────────────────────────────────────────────────────────────────
# En esta sección se definen funciones auxiliares que se usan durante la simulación.


def fmt(seg: float) -> str:
    """Convierte segundos desde las 00:00 a formato HH:MM:SS."""
    s = int(round(seg))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def normal_pos(media: float, desv: float, rng: random.Random):
    """
    Genera una variable aleatoria normal positiva.

    Se usa el método de Box-Muller:
    - r1 y r2 son los números aleatorios usados.
    - z es una normal estándar.
    - v = media + desvío * z.

    Como el enunciado pide tiempos normales mayores a cero, si el valor generado
    es negativo o cero se descarta y se vuelve a generar.
    """
    while True:
        r1, r2 = rng.random(), rng.random()
        z = math.sqrt(-2 * math.log(r1)) * math.cos(2 * math.pi * r2)
        v = media + desv * z
        if v > 0:
            return r1, r2, v


def uniforme(a: float, b: float, rng: random.Random):
    """
    Genera una variable aleatoria uniforme continua entre a y b.

    Devuelve:
    - r: número aleatorio usado.
    - valor generado entre a y b.
    """
    r = rng.random()
    return r, a + r * (b - a)


def unif_disc(a: int, b: int, rng: random.Random):
    """
    Genera una variable aleatoria uniforme discreta entera entre a y b.

    Se usa para obtener el valor de A en los pedidos de consumo local.
    En el enunciado, A es uniforme discreto entre 2 y 5.
    """
    r = rng.random()
    return r, min(a + int(r * (b - a + 1)), b)


def permanencia_params(salon: str, reloj: float):
    """
    Devuelve la media y el desvío, en segundos, del tiempo de permanencia
    según el salón y la franja horaria.

    El reloj está en segundos. Por eso se divide por 3600 para obtener la hora.
    """
    h = reloj / 3600

    # Cada tupla representa: hora_desde, hora_hasta, media_minutos, desvío_minutos.
    tabla = {
        "rojo":  [(11, 12, 20, 15), (12, 13, 30, 15), (13, 14, 35, 15), (14, 15, 20, 15)],
        "azul":  [(11, 12, 30, 15), (12, 13, 40, 15), (13, 14, 45, 10), (14, 15, 35, 15)],
    }

    # Busca la franja horaria correspondiente al reloj actual.
    for (h0, h1, med, dev) in tabla[salon]:
        if h0 <= h < h1:
            return med * 60, dev * 60

    # Valor de respaldo por si el reloj queda fuera de las franjas esperadas.
    return 20 * 60, 15 * 60


def rk_local(a_val: int, h: float = 0.01):
    """
    Calcula el tiempo de preparación de un pedido para consumo local mediante Runge-Kutta.

    Ecuación diferencial del enunciado:
        dL/dt - 3A = 6
    Entonces:
        dL/dt = 6 + 3A

    Condiciones:
    - A toma un valor uniforme discreto entre 2 y 5.
    - L inicial es igual a A.
    - h = 0.01, modificable desde la interfaz.
    - El pedido está listo cuando L supera 10.
    - t = 1 equivale a 10 minutos, por eso al final se multiplica por 600 segundos.
    """
    def f(t, l):
        # En esta ecuación, la derivada no depende de t ni de L, sólo de A.
        return 6 + 3 * a_val

    # t es el tiempo adimensional de RK y l es el valor de L.
    t, l = 0.0, float(a_val)

    # rows guarda la tabla completa de Runge-Kutta para poder mostrarla en Streamlit.
    rows = []

    # Avanza hasta que L supere 10.
    while l <= 10:
        k1 = f(t, l)
        k2 = f(t + h/2, l + h*k1/2)
        k3 = f(t + h/2, l + h*k2/2)
        k4 = f(t + h, l + h*k3)

        # Fórmula de Runge-Kutta de cuarto orden.
        l2 = l + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t2 = t + h

        # Guarda una fila de la tabla de RK.
        rows.append({
            "A": a_val,
            "t": round(t, 4),
            "L": round(l, 4),
            "k1": round(k1, 4),
            "k2": round(k2, 4),
            "k3": round(k3, 4),
            "k4": round(k4, 4),
            "t_sig": round(t2, 4),
            "L_sig": round(l2, 4),
        })

        # Actualiza t y L para el próximo paso.
        t, l = t2, l2

    # Como t = 1 equivale a 10 minutos, se convierte t a segundos multiplicando por 600.
    return t * 600, rows


# ── simulación ────────────────────────────────────────────────────────────────

# Diccionario usado para cambiar nombres internos de columnas por nombres más claros
# al momento de mostrar el vector de estado.
RENAME = {
    "iteracion": "Iteracion",
    "evento": "Evento",
    "reloj_seg": "Reloj (seg)",
    "hora": "Hora",
    "r1_llg": "RND 1 llegada",
    "r2_llg": "RND 2 llegada",
    "tel": "Tiempo entre llegadas",
    "prox_llg": "Proxima Llegada",
    "r_tipo": "RND tipo",
    "tipo_pedido": "Tipo de Pedido",
    "r_caja": "RND atencion",
    "ta_caja": "Tiempo atencion",
    "fin_caja": "Fin Atencion Caja",
    "r_A": "RND CL",
    "A": "Valor de A",
    "tp_prep": "Tiempo de Preparacion CL",
    "fin_prep_local": "Fin Preparacion CL",
    "r1_llevar": "RND1 Para Llevar",
    "r2_llevar": "RND2 Para Llevar",
    "tp_llevar": "Tiempo Preparacion Llevar",
    "fin_prep_llevar": "Fin Preparacion Llevar",
    "r_salon": "RND Salon",
    "salon_elegido": "Salon",
    "r_sal1": "RND1 Permanencia",
    "r_sal2": "RND2 Permanencia",
    "tp_sal": "Tiempo Permanencia Salon",
    "fin_sal": "Fin Permanencia Salon",
    "caja_est": "Estado Caja",
    "caja_ini": "Hr Inicio Ocupacion",
    "caja_ac": "AC Tiempo Ocupado",
    "cola_caja": "Cola Caja",
    "max_cola_caja": "MAX Cola Caja",
    "m1_est": "M1 Estado", "m2_est": "M2 Estado", "m3_est": "M3 Estado",
    "m1_ini": "M1 Hr Inicio", "m2_ini": "M2 Hr Inicio", "m3_ini": "M3 Hr Inicio",
    "m1_ac": "M1 AC", "m2_ac": "M2 AC", "m3_ac": "M3 AC",
    "cola_most": "Cola Mostrador",
    "max_cola_most": "MAX Cola Mostrador",
    "r_ocup": "Salon Rojo Ocupados",
    "r_esp": "Salon Rojo Esperando",
    "r_ac_lleno": "AC Salon Rojo Lleno",
    "a_ocup": "Salon Azul Ocupados",
    "a_esp": "Salon Azul Esperando",
    "a_ac_lleno": "AC Salon Azul Lleno",
    "ac_perm": "AC Permanencia", "n_perm": "N Permanencia",
    "ac_cc": "AC Cola Caja", "n_cc": "N Cola Caja",
    "ac_cm": "AC Cola Mostrador", "n_cm": "N Cola Mostrador",
    "n_llevar": "N Para Llevar", "n_local": "N Local",
    "n_rojo": "N Salon Rojo", "n_azul": "N Salon Azul",
    "vivos": "Clientes Vivos",
}

CLI_FIELD_LABELS = {
    "id": "ID", "est": "Estado", "llg": "Hora Llegada",
    "t_cc": "T Cola Caja", "perm": "Permanencia",
}


def simular(p: dict) -> dict:
    """
    Ejecuta la simulación completa.

    Recibe un diccionario p con todos los parámetros cargados desde la interfaz.
    Devuelve otro diccionario con:
    - vector de estado,
    - controles periódicos,
    - tablas Runge-Kutta,
    - métricas finales.
    """

    # Generador aleatorio con semilla fija para que la corrida sea reproducible.
    rng = random.Random(p["semilla"])

    # Variables principales de reloj y control de iteraciones.
    reloj = p["hora_inicio"]
    fin = p["hora_fin"]
    it = 0

    # Identificador incremental para cada cliente que llega al sistema.
    nxt_id = 1

    # -------------------------------------------------------------------------
    # Estado de la caja
    # -------------------------------------------------------------------------
    # caja_est: libre u ocupado.
    # caja_cli: id del cliente actualmente atendido.
    # caja_ini: momento en que empezó la atención actual.
    # caja_ac: acumulador de tiempo ocupado de caja.
    caja_est = "libre"
    caja_cli = None
    caja_ini = None
    caja_ac = 0.0

    # Cola FIFO de caja y máximo observado de cola.
    cola_caja: deque = deque()
    max_cc = 0

    # -------------------------------------------------------------------------
    # Estado del mostrador
    # -------------------------------------------------------------------------
    # Hay 3 empleados en mostrador.
    # Cada empleado tiene:
    # - est: libre u ocupado.
    # - cli: cliente que está preparando.
    # - ini: inicio de preparación.
    # - fin: próximo fin de preparación.
    # - ac: acumulador de tiempo ocupado de ese empleado.
    mostr = [
        {"est": "libre", "cli": None, "ini": None, "fin": INF, "ac": 0.0}
        for _ in range(3)
    ]

    # Cola FIFO del mostrador y máximo observado de cola.
    cola_most: deque = deque()
    max_cm = 0

    # -------------------------------------------------------------------------
    # Estado de los salones
    # -------------------------------------------------------------------------
    # salon_r y salon_a guardan los ids de clientes actualmente dentro de cada salón.
    salon_r: set = set()
    salon_a: set = set()

    # esp_r y esp_a representan esperas para ingresar a salón cuando está lleno.
    esp_r: deque = deque()
    esp_a: deque = deque()

    # salida_sal guarda, para cada cliente en salón, su hora programada de salida.
    salida_sal: dict = {}

    # Variables para calcular cuánto tiempo estuvo lleno cada salón.
    salon_r_lleno_ini: Optional[float] = None
    salon_a_lleno_ini: Optional[float] = None
    ac_r_lleno = 0.0
    ac_a_lleno = 0.0

    # -------------------------------------------------------------------------
    # Estructuras generales de datos
    # -------------------------------------------------------------------------
    # clientes guarda todos los clientes vivos en el sistema.
    # Cuando un cliente se va, se elimina de este diccionario.
    clientes: Dict[int, dict] = {}

    # vector guarda una fila por cada evento ocurrido.
    vector: List[dict] = []

    # rk_rows guarda todas las filas de Runge-Kutta generadas para pedidos locales.
    rk_rows: List[dict] = []

    # Controles periódicos solicitados por el enunciado.
    ctrl_most: List[dict] = []
    ctrl_sal: List[dict] = []

    # -------------------------------------------------------------------------
    # Acumuladores para métricas
    # -------------------------------------------------------------------------
    # ac_perm/n_perm: permanencia promedio en el negocio.
    # ac_cc/n_cc: tiempo promedio en cola de caja.
    # ac_cm/n_cm: tiempo promedio en cola de mostrador.
    ac_perm = 0.0
    n_perm = 0
    ac_cc = 0.0
    n_cc = 0
    ac_cm = 0.0
    n_cm = 0

    # Contadores de tipos de pedido y salón elegido.
    n_llevar = 0
    n_local = 0
    n_rojo = 0
    n_azul = 0

    # -------------------------------------------------------------------------
    # Inicialización de próximos eventos
    # -------------------------------------------------------------------------
    # Se genera la primera llegada antes de tomar la fila inicial del vector.
    r1, r2, tel = normal_pos(p["media_llg"], p["desv_llg"], rng)
    prox_llg = reloj + tel

    # Próximos controles periódicos:
    # - cola del mostrador cada ctrl_most segundos.
    # - ocupación de salones cada ctrl_sal segundos.
    prox_ctrl_m = p["hora_inicio"] + p["ctrl_most"]
    prox_ctrl_s = p["hora_inicio"] + p["ctrl_sal"]

    last: dict = {"r1_llg": r1, "r2_llg": r2, "tel": tel}

    def libre_most():
        """Devuelve el índice del primer empleado libre del mostrador, o None si no hay."""
        for i, e in enumerate(mostr):
            if e["est"] == "libre":
                return i
        return None

    def ini_caja(cid, ahora):
        """
        Inicia la atención en caja para un cliente.

        También:
        - calcula su tiempo de espera en cola de caja,
        - genera el tiempo de atención uniforme,
        - programa el fin de atención en caja,
        - actualiza el estado de caja.
        """
        nonlocal caja_est, caja_cli, caja_ini, ac_cc, n_cc

        c = clientes[cid]
        c["est"] = "en_caja"
        c["ini_caja"] = ahora

        # Tiempo en cola de caja = momento en que empieza atención - momento de llegada.
        c["t_cc"] = ahora - c["llg"]
        ac_cc += c["t_cc"]
        n_cc += 1

        # Tiempo de atención en caja: uniforme entre caja_a y caja_b.
        r, ta = uniforme(p["caja_a"], p["caja_b"], rng)
        c["fin_caja"] = ahora + ta

        # Actualización del recurso caja.
        caja_est = "ocupado"
        caja_cli = cid
        caja_ini = ahora

        # Guarda los valores aleatorios para que salgan en la fila del vector de estado.
        last.update({"r_caja": r, "ta_caja": ta})

    def ini_prep(cid, ei, ahora):
        """
        Inicia la preparación del pedido en un empleado del mostrador.

        Parámetros:
        - cid: id del cliente.
        - ei: índice del empleado de mostrador.
        - ahora: reloj actual.
        """
        nonlocal ac_cm, n_cm

        c = clientes[cid]
        c["est"] = "prep"
        c["ini_prep"] = ahora

        # Tiempo en cola del mostrador = inicio de preparación - llegada al mostrador.
        c["t_cm"] = ahora - c["llg_most"]
        ac_cm += c["t_cm"]
        n_cm += 1

        if c["tipo"] == "llevar":
            r1l, r2l, tp = normal_pos(p["llevar_a"], p["llevar_b"], rng)
            fin_pl = ahora + tp
            last.update({"r1_llevar": r1l, "r2_llevar": r2l, "tp_llevar": tp, "fin_prep_llevar": round(fin_pl, 2)})
        else:
            # Preparación para consumo local: se genera A y luego se calcula RK.
            ra, av = unif_disc(2, 5, rng)
            tp, rk = rk_local(av, p["h_rk"])

            # Se agregan las filas de esta tabla RK al acumulado general.
            rk_rows.extend(rk)

            # Se guardan RND, A, tiempo de preparación y fin de preparación local.
            last.update({
                "r_A": ra,
                "A": av,
                "tp": tp,
                "fin_prep_local": round(ahora + tp, 2),
            })

        # Programa el fin de preparación y ocupa al empleado correspondiente.
        c["fin_prep"] = ahora + tp
        mostr[ei].update({"est": "ocupado", "cli": cid, "ini": ahora, "fin": c["fin_prep"]})

    def intentar_salon(cid, ahora):
        """
        Intenta ubicar a un cliente local en su salón elegido.

        Si hay capacidad, entra al salón.
        Si no hay capacidad, queda esperando en la cola del salón correspondiente.
        """
        c = clientes[cid]

        if c["salon"] == "rojo":
            if len(salon_r) < p["cap_r"]:
                ingresar_salon(cid, ahora)
            else:
                c["est"] = "esp_r"
                esp_r.append(cid)
        else:
            if len(salon_a) < p["cap_a"]:
                ingresar_salon(cid, ahora)
            else:
                c["est"] = "esp_a"
                esp_a.append(cid)

    def ingresar_salon(cid, ahora):
        """
        Hace ingresar a un cliente a su salón.

        También genera su tiempo de permanencia según salón y franja horaria,
        y programa su evento de salida del salón.
        """
        nonlocal salon_r_lleno_ini, salon_a_lleno_ini, ac_r_lleno, ac_a_lleno

        c = clientes[cid]
        c["est"] = "en_salon"
        c["ini_salon"] = ahora

        # Obtiene media y desvío de permanencia según salón y hora actual.
        med, dev = permanencia_params(c["salon"], ahora)

        # Tiempo de permanencia en salón: normal positiva.
        r1s, r2s, tp = normal_pos(med, dev, rng)
        c["fin_salon"] = ahora + tp

        # Registra el próximo evento de salida de salón para este cliente.
        salida_sal[cid] = c["fin_salon"]
        last.update({"r_sal1": r1s, "r_sal2": r2s, "tp_sal": tp, "fin_sal": round(c["fin_salon"], 2)})
        if c["salon"] == "rojo":
            salon_r.add(cid)

            # Si el salón acaba de quedar lleno, se guarda desde cuándo está lleno.
            if len(salon_r) >= p["cap_r"] and salon_r_lleno_ini is None:
                salon_r_lleno_ini = ahora
        else:
            salon_a.add(cid)

            # Si el salón acaba de quedar lleno, se guarda desde cuándo está lleno.
            if len(salon_a) >= p["cap_a"] and salon_a_lleno_ini is None:
                salon_a_lleno_ini = ahora

    def cerrar(cid, ahora):
        """
        Cierra la vida de un cliente en el sistema.

        Se usa cuando:
        - termina la preparación de un pedido para llevar,
        - o un cliente que consumió en salón se retira.
        """
        nonlocal ac_perm, n_perm, salon_r_lleno_ini, salon_a_lleno_ini, ac_r_lleno, ac_a_lleno

        # Se elimina el cliente de clientes vivos.
        c = clientes.pop(cid)
        c["est"] = "fue"
        c["salida"] = ahora

        # Permanencia total en el negocio.
        c["perm"] = ahora - c["llg"]
        ac_perm += c["perm"]
        n_perm += 1

        # Control de tiempo acumulado con salón rojo lleno.
        # Si estaba lleno y luego de la salida queda con espacio, termina el período lleno.
        if c.get("salon") == "rojo" and salon_r_lleno_ini is not None and len(salon_r) < p["cap_r"]:
            ac_r_lleno += ahora - salon_r_lleno_ini
            salon_r_lleno_ini = None

        # Control de tiempo acumulado con salón azul lleno.
        if c.get("salon") == "azul" and salon_a_lleno_ini is not None and len(salon_a) < p["cap_a"]:
            ac_a_lleno += ahora - salon_a_lleno_ini
            salon_a_lleno_ini = None

    def snap(evento):
        """
        Toma una fotografía del sistema en el instante actual.

        Esta función construye una fila del vector de estado.
        Incluye:
        - evento actual,
        - reloj,
        - RNDs usados en el evento,
        - próximos eventos,
        - estado de caja,
        - estado de mostrador,
        - estado de salones,
        - acumuladores de métricas,
        - algunos clientes vivos.
        """
        row = {
            # Datos generales de la fila.
            "iteracion": it,
            "evento": evento,
            "reloj_seg": round(reloj, 2),
            "hora": fmt(reloj),

            # Llegadas: RNDs y tiempo entre llegadas generado.
            "r1_llg": last.get("r1_llg"),
            "r2_llg": last.get("r2_llg"),
            "tel": last.get("tel"),
            "prox_llg": round(prox_llg, 2) if prox_llg < INF else None,

            # Tipo de pedido: para llevar o local.
            "r_tipo": last.get("r_tipo"),
            "tipo_pedido": last.get("tipo_pedido"),

            # Caja: RND, tiempo de atención y próximo fin de atención.
            "r_caja": last.get("r_caja"),
            "ta_caja": last.get("ta_caja"),
            "fin_caja": clientes[caja_cli]["fin_caja"] if caja_cli in clientes else None,
            # rk / llevar
            "r_A": last.get("r_A"), "A": last.get("A"),
            "r1_llevar": last.get("r1_llevar"), "r2_llevar": last.get("r2_llevar"),
            "tp_prep": last.get("tp"),
            "fin_prep_local": last.get("fin_prep_local"),
            "tp_llevar": last.get("tp_llevar"),
            "fin_prep_llevar": last.get("fin_prep_llevar"),
            "salon_elegido": last.get("salon_elegido"),

            # Elección y permanencia en salón.
            "r_salon": last.get("r_salon"),
            "r_sal1": last.get("r_sal1"),
            "r_sal2": last.get("r_sal2"),
            "tp_sal": last.get("tp_sal"),
            "fin_sal": last.get("fin_sal"),
            # caja estado
            "caja_est": caja_est, "caja_cli": caja_cli,
            "caja_ini": None if caja_ini is None else round(caja_ini, 2),
            "cola_caja": len(cola_caja), "max_cola_caja": max_cc,
            "caja_ac": round(caja_ac,2),
            # mostrador
            **{f"m{i+1}_est":   mostr[i]["est"]               for i in range(3)},
            **{f"m{i+1}_cli":   mostr[i]["cli"]               for i in range(3)},
            **{f"m{i+1}_ini":   None if mostr[i]["ini"] is None else round(mostr[i]["ini"],2) for i in range(3)},
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
        # clientes vivos
        for k, cv in enumerate(clientes.values(), 1):
            row[f"cli{k}_id"]   = cv["id"]
            row[f"cli{k}_est"]  = cv["est"]
            row[f"cli{k}_llg"]  = fmt(cv["llg"])
            row[f"cli{k}_t_cc"] = round(cv.get("t_cc", 0), 2)
            row[f"cli{k}_perm"] = round(cv.get("perm", 0), 2)
        vector.append(row)

    # -------------------------------------------------------------------------
    # Fila inicial del vector de estado
    # -------------------------------------------------------------------------
    # Representa el estado del sistema antes de que ocurra el primer evento.
    # Ya tiene programada la primera llegada y sus RNDs correspondientes.
    snap("inicializacion")

    # -------------------------------------------------------------------------
    # Bucle principal de simulación
    # -------------------------------------------------------------------------
    # En cada iteración:
    # 1) Se determina cuál es el próximo evento.
    # 2) Se avanza el reloj a ese instante.
    # 3) Se ejecuta la lógica del evento.
    # 4) Se guarda una nueva fila del vector de estado.
    while it < p["max_it"]:
        # Limpia los datos aleatorios del evento anterior.
        # Sólo se cargarán en last los RNDs usados en el evento actual.
        last = {}

        # Próximo fin de caja, si hay cliente en caja.
        fc = clientes[caja_cli]["fin_caja"] if caja_cli in clientes else INF

        # Próximo fin de preparación entre los 3 empleados del mostrador.
        fp = min(m["fin"] for m in mostr)

        # Próxima salida de salón, si hay clientes en algún salón.
        fs = min(salida_sal.values()) if salida_sal else INF

        # Diccionario con todos los posibles próximos eventos y sus tiempos.
        times = {
            "llegada": prox_llg,
            "fin_caja": fc,
            "fin_prep": fp,
            "salida_salon": fs,
            "ctrl_most": prox_ctrl_m,
            "ctrl_sal": prox_ctrl_s,
            "fin_sim": fin,
        }

        # Selecciona el evento con menor tiempo programado.
        ev = min(times, key=times.get)
        reloj = times[ev]
        it += 1

        # Si el próximo evento ocurre después del fin de simulación, se corta en fin_sim.
        if reloj >= fin:
            reloj = fin
            ev = "fin_sim"

        if ev == "llegada":
            # -----------------------------------------------------------------
            # Evento: llegada de cliente
            # -----------------------------------------------------------------
            # Se crea un nuevo cliente en estado cola_caja.
            c = {
                "id": nxt_id,
                "llg": reloj,
                "est": "cola_caja",
                "tipo": None,
                "salon": None,
                "ini_caja": None,
                "fin_caja": None,
                "llg_most": None,
                "t_cc": 0.0,
                "t_cm": 0.0,
                "perm": 0.0,
            }
            clientes[nxt_id] = c

            # Si la caja está libre, el cliente pasa directamente a caja.
            # Si está ocupada, espera en la cola de caja.
            if caja_est == "libre":
                ini_caja(nxt_id, reloj)
            else:
                cola_caja.append(nxt_id)
                max_cc = max(max_cc, len(cola_caja))

            nxt_id += 1

            # Se programa la próxima llegada.
            r1, r2, tel = normal_pos(p["media_llg"], p["desv_llg"], rng)
            prox_llg = reloj + tel
            last.update({"r1_llg": r1, "r2_llg": r2, "tel": tel})

        elif ev == "fin_caja":
            # -----------------------------------------------------------------
            # Evento: fin de atención en caja
            # -----------------------------------------------------------------
            # El cliente que estaba en caja pasa al mostrador.
            nonlocal_cid = caja_cli
            c = clientes[nonlocal_cid]
            c["est"] = "cola_most"
            c["llg_most"] = reloj

            # Se determina si compra para llevar o para consumir en local.
            rt = rng.random()
            if rt < p["prob_llevar"]:
                c["tipo"] = "llevar"
                n_llevar += 1
                last.update({"r_tipo": rt, "tipo_pedido": "llevar"})
            else:
                c["tipo"] = "local"
                n_local += 1

                # Si consume en local, se determina salón elegido.
                rs = rng.random()
                c["salon"] = "rojo" if rs < p["prob_rojo"] else "azul"

                if c["salon"] == "rojo":
                    n_rojo += 1
                else:
                    n_azul += 1

                last.update({
                    "r_tipo": rt,
                    "tipo_pedido": "local",
                    "r_salon": rs,
                    "salon_elegido": c["salon"],
                })

            # Se acumula el tiempo ocupado de caja para calcular ocupación.
            if caja_ini is not None:
                caja_ac += reloj - caja_ini

            # Si hay cola de caja, entra el siguiente cliente.
            # Si no hay cola, la caja queda libre.
            if cola_caja:
                sig = cola_caja.popleft()
                ini_caja(sig, reloj)
            else:
                caja_est = "libre"
                caja_cli = None
                caja_ini = None

            # El cliente que salió de caja intenta iniciar preparación en mostrador.
            ei = libre_most()
            if ei is not None:
                ini_prep(nonlocal_cid, ei, reloj)
            else:
                cola_most.append(nonlocal_cid)
                max_cm = max(max_cm, len(cola_most))

        elif ev == "fin_prep":
            # -----------------------------------------------------------------
            # Evento: fin de preparación en mostrador
            # -----------------------------------------------------------------
            # Se identifica qué empleado terminó primero.
            fins = [m["fin"] for m in mostr]
            ei = fins.index(min(fins))
            cid = mostr[ei]["cli"]

            # Se acumula el tiempo ocupado de ese empleado.
            if mostr[ei]["ini"] is not None:
                mostr[ei]["ac"] += reloj - mostr[ei]["ini"]

            # El empleado queda libre.
            mostr[ei].update({"est": "libre", "cli": None, "ini": None, "fin": INF})

            c = clientes[cid]

            # Si el pedido era para llevar, el cliente se va.
            # Si era local, intenta ingresar al salón elegido.
            if c["tipo"] == "llevar":
                cerrar(cid, reloj)
            else:
                intentar_salon(cid, reloj)

            # Si había cola de mostrador, el empleado toma al siguiente cliente.
            if cola_most:
                sig = cola_most.popleft()
                ini_prep(sig, ei, reloj)

        elif ev == "salida_salon":
            # -----------------------------------------------------------------
            # Evento: salida de un cliente del salón
            # -----------------------------------------------------------------
            # Se busca el cliente con menor hora de salida programada.
            cid = min(salida_sal, key=salida_sal.get)
            salida_sal.pop(cid)

            c = clientes.get(cid)
            if c:
                s = c["salon"]

                # Se lo elimina del conjunto de ocupados del salón correspondiente.
                (salon_r if s == "rojo" else salon_a).discard(cid)

                # Se cierra su permanencia en el sistema.
                cerrar(cid, reloj)

                # Si había clientes esperando para ese salón, entra el primero de la cola.
                q = esp_r if s == "rojo" else esp_a
                if q:
                    ingresar_salon(q.popleft(), reloj)

        elif ev == "ctrl_most":
            # -----------------------------------------------------------------
            # Evento: control periódico de cola de mostrador
            # -----------------------------------------------------------------
            # Guarda cuánta gente hay en cola frente al mostrador en este instante.
            ctrl_most.append({"reloj": round(reloj, 2), "hora": fmt(reloj), "cola_most": len(cola_most)})

            # Programa el siguiente control.
            prox_ctrl_m += p["ctrl_most"]

        elif ev == "ctrl_sal":
            # -----------------------------------------------------------------
            # Evento: control periódico de salones
            # -----------------------------------------------------------------
            # Guarda cuántas personas hay en cada salón en este instante.
            ctrl_sal.append({
                "reloj": round(reloj, 2),
                "hora": fmt(reloj),
                "salon_rojo": len(salon_r),
                "salon_azul": len(salon_a),
            })

            # Programa el siguiente control.
            prox_ctrl_s += p["ctrl_sal"]

        # Después de ejecutar el evento, se guarda la fila del vector de estado.
        snap(ev)

        # Si se llegó al fin de simulación, termina el bucle.
        if ev == "fin_sim":
            break

    # -------------------------------------------------------------------------
    # Cálculo final de métricas
    # -------------------------------------------------------------------------
    dur = fin - p["hora_inicio"]

    # Tiempo ocupado total de los tres empleados de mostrador.
    ac_m = sum(m["ac"] for m in mostr)

    metricas = {
        "Prom. permanencia negocio (seg)": ac_perm / n_perm if n_perm else 0,
        "Prom. tiempo cola caja (seg)": ac_cc / n_cc if n_cc else 0,
        "Prom. tiempo cola mostrador (seg)": ac_cm / n_cm if n_cm else 0,
        "% Ocupación caja": caja_ac / dur * 100,
        "% Ocupación empleados mostrador": ac_m / (3 * dur) * 100,
        "Máx. cola caja": max_cc,
        "Máx. cola mostrador": max_cm,
        "AC tiempo salón rojo lleno (seg)": ac_r_lleno,
        "AC tiempo salón azul lleno (seg)": ac_a_lleno,
        "Clientes finalizados": n_perm,
        "Para llevar": n_llevar,
        "En local": n_local,
        "Salón rojo": n_rojo,
        "Salón azul": n_azul,
    }

    # Se convierte el vector a DataFrame y se renombran columnas para que sean más claras.
    df_vector = pd.DataFrame(vector).rename(columns=RENAME)
    cli_rename = {}
    for col in df_vector.columns:
        m = re.match(r"cli(\d+)_(id|est|llg|t_cc|perm)", col)
        if m:
            k, campo = m.groups()
            cli_rename[col] = f"C{k} {CLI_FIELD_LABELS[campo]}"
    df_vector = df_vector.rename(columns=cli_rename)
    return {
        "vector": df_vector,
        "ctrl_most": pd.DataFrame(ctrl_most) if ctrl_most else pd.DataFrame(),
        "ctrl_sal":  pd.DataFrame(ctrl_sal)  if ctrl_sal  else pd.DataFrame(),
        "rk":        pd.DataFrame(rk_rows)   if rk_rows   else pd.DataFrame(),
        "metricas":  metricas,
    }

# ── tabla multinivel ───────────────────────────────────────────────────────────

GRUPOS = [
    ("", ["Iteracion", "Evento", "Reloj (seg)", "Hora"]),
    ("Proxima Llegada", ["RND 1 llegada", "RND 2 llegada", "Tiempo entre llegadas", "Proxima Llegada"]),
    ("Fin Atencion Caja", ["RND tipo", "Tipo de Pedido", "RND atencion", "Tiempo atencion", "Fin Atencion Caja"]),
    ("Consumo Local", ["RND CL", "Valor de A", "Tiempo de Preparacion CL", "Fin Preparacion CL"]),
    ("Para Llevar", ["RND1 Para Llevar", "RND2 Para Llevar", "Tiempo Preparacion Llevar", "Fin Preparacion Llevar"]),
    ("Permanencia Salon", ["RND Salon", "Salon", "RND1 Permanencia", "RND2 Permanencia", "Tiempo Permanencia Salon", "Fin Permanencia Salon"]),
    ("Empleado Caja", ["Estado Caja", "Hr Inicio Ocupacion", "AC Tiempo Ocupado", "Cola Caja", "MAX Cola Caja"]),
    ("Mostrador 1", ["M1 Estado", "M1 Hr Inicio", "M1 AC"]),
    ("Mostrador 2", ["M2 Estado", "M2 Hr Inicio", "M2 AC"]),
    ("Mostrador 3", ["M3 Estado", "M3 Hr Inicio", "M3 AC"]),
    ("Cola Mostrador", ["Cola Mostrador", "MAX Cola Mostrador"]),
    ("Salon Rojo", ["Salon Rojo Ocupados", "Salon Rojo Esperando", "AC Salon Rojo Lleno"]),
    ("Salon Azul", ["Salon Azul Ocupados", "Salon Azul Esperando", "AC Salon Azul Lleno"]),
    ("Acumuladores", ["AC Permanencia", "N Permanencia", "AC Cola Caja", "N Cola Caja", "AC Cola Mostrador", "N Cola Mostrador", "N Para Llevar", "N Local", "N Salon Rojo", "N Salon Azul"]),
]


def grupos_con_clientes(df: pd.DataFrame) -> list:
    cli_nums = sorted({int(m.group(1)) for c in df.columns if (m := re.match(r"C(\d+) ID", c))})
    grupos = list(GRUPOS)
    for k in cli_nums:
        grupos.append((f"Cliente {k}", [f"C{k} ID", f"C{k} Estado", f"C{k} Hora Llegada", f"C{k} T Cola Caja", f"C{k} Permanencia"]))
    return grupos


def render_tabla_multinivel(df: pd.DataFrame, grupos: list) -> str:
    header1 = ""
    header2 = ""
    for grupo, subcols in grupos:
        presentes = [c for c in subcols if c in df.columns]
        if not presentes:
            continue
        header1 += f'<th colspan="{len(presentes)}" style="text-align:center;border:1px solid #444;padding:4px;background:#2d2d2d">{grupo}</th>'
        for sc in presentes:
            header2 += f'<th style="text-align:center;border:1px solid #444;padding:4px;white-space:pre-wrap;max-width:80px;font-size:0.75em">{sc.replace(" ", "<br>")}</th>'

    rows = ""
    for _, row in df.iterrows():
        rows += "<tr>"
        for grupo, subcols in grupos:
            for sc in subcols:
                if sc in df.columns:
                    val = row.get(sc, "")
                    rows += f'<td style="text-align:center;border:1px solid #333;padding:3px;font-size:0.8em">{val}</td>'
        rows += "</tr>"

    html = f"""
    <div style="overflow-x:auto">
    <table style="border-collapse:collapse;width:100%;font-family:monospace">
        <thead>
            <tr>{header1}</tr>
            <tr>{header2}</tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    </div>
    """
    return html

# ── UI ────────────────────────────────────────────────────────────────────────
# A partir de acá se construye la interfaz visual con Streamlit.

# Título principal de la aplicación.
st.title("🥩 Simulación Lomitería – TP5 Grupo 22")

# Panel lateral donde el usuario carga los parámetros de la simulación.
with st.sidebar:
    st.header("⚙️ Parámetros")

    # Semilla usada para reproducir la misma secuencia de números aleatorios.
    semilla = st.number_input("Semilla aleatoria", value=22, step=1)

    st.subheader("Tiempo simulación")
    # Horas de inicio y fin de la simulación. Luego se convierten a segundos.
    h_ini = st.number_input("Hora inicio", 0, 23, 11)
    h_fin = st.number_input("Hora fin", 0, 24, 15)

    st.subheader("Llegada de clientes (Normal)")
    # Parámetros de la distribución normal positiva de llegadas.
    media_llg = st.number_input("Media (seg)", value=60.0, min_value=1.0)
    desv_llg = st.number_input("Desvío (seg)", value=30.0, min_value=0.1)

    st.subheader("Atención en caja (Uniforme)")
    # Parámetros de la atención en caja: uniforme entre mínimo y máximo.
    caja_a = st.number_input("Mín caja (seg)", value=15.0, min_value=0.0)
    caja_b = st.number_input("Máx caja (seg)", value=45.0, min_value=0.1)
    if caja_b <= caja_a:
        st.warning("Máx debe ser > Mín")

    st.subheader("Tipo de pedido")
    # Probabilidad de pedido para llevar. El resto consume en local.
    prob_llevar = st.slider("% Para llevar", 0, 100, 25) / 100

    st.subheader("Preparación para llevar (Normal)")
    llevar_a = st.number_input("Media llevar (seg)", value=100.0, min_value=0.0)
    llevar_b = st.number_input("Desvío llevar (seg)", value=20.0, min_value=0.1)

    st.subheader("Runge-Kutta (consumo local)")
    # Paso h para la resolución numérica de la ecuación diferencial.
    h_rk = st.number_input("Paso h", value=0.01, min_value=0.0001, format="%.4f")

    st.subheader("Salones")
    # Probabilidad de elegir salón rojo y capacidades de salones.
    prob_rojo = st.slider("% Salón Rojo", 0, 100, 30) / 100
    cap_r = st.number_input("Capacidad salón Rojo", value=30, min_value=1)
    cap_a = st.number_input("Capacidad salón Azul", value=40, min_value=1)

    st.subheader("Controles periódicos")
    # Frecuencia de controles periódicos, expresada en minutos en la interfaz.
    ctrl_most_min = st.number_input("Cada N min → cola mostrador", value=15, min_value=1)
    ctrl_sal_min = st.number_input("Cada N min → salones", value=30, min_value=1)

    st.subheader("Iteraciones")
    # Límite máximo de iteraciones de la simulación.
    max_it = st.number_input("Máx iteraciones", value=100000, min_value=1, step=1000)

    st.subheader("Filtro vector estado")
    # Permite mostrar i filas del vector a partir de una iteración j.
    j_it = st.number_input("Desde iteración j", value=0, min_value=0, step=1)
    i_it = st.number_input("Mostrar i iteraciones", value=50, min_value=1, step=10)

    # Botón que ejecuta la simulación.
    correr = st.button("▶ Simular", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Validaciones básicas de parámetros
# -----------------------------------------------------------------------------
errores = []
if h_fin <= h_ini: errores.append("Hora fin debe ser mayor que hora inicio.")
if caja_b <= caja_a: errores.append("Máx caja debe ser > Mín caja.")

# Si hay errores, se muestran y se detiene la app.
if errores:
    for e in errores:
        st.error(e)
    st.stop()

# -----------------------------------------------------------------------------
# Ejecución de la simulación al presionar el botón
# -----------------------------------------------------------------------------
if correr:
    # Se construye el diccionario de parámetros para simular().
    # Las horas se convierten a segundos.
    params = dict(
        semilla=int(semilla),
        hora_inicio=int(h_ini) * 3600,
        hora_fin=int(h_fin) * 3600,
        media_llg=media_llg,
        desv_llg=desv_llg,
        caja_a=caja_a,
        caja_b=caja_b,
        prob_llevar=prob_llevar,
        llevar_a=llevar_a,
        llevar_b=llevar_b,
        h_rk=h_rk,
        prob_rojo=prob_rojo,
        cap_r=int(cap_r),
        cap_a=int(cap_a),
        ctrl_most=ctrl_most_min * 60,
        ctrl_sal=ctrl_sal_min * 60,
        max_it=int(max_it),
    )

    # Muestra un spinner mientras corre la simulación.
    with st.spinner("Simulando..."):
        res = simular(params)

    # Guarda los resultados en session_state para que no se pierdan al cambiar pestañas.
    st.session_state["res"] = res
    st.session_state["j"] = int(j_it)
    st.session_state["i"] = int(i_it)

    st.success(f"Simulación completada – {len(res['vector'])} iteraciones")

# Si todavía no se simuló, se muestra un mensaje inicial y se corta la ejecución.
if "res" not in st.session_state:
    st.info("Configurá los parámetros en el panel izquierdo y presioná **Simular**.")
    st.stop()

# Recupera los resultados guardados.
res: dict = st.session_state["res"]
j = st.session_state["j"]
i = st.session_state["i"]

# Crea las pestañas de visualización.
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Vector de Estado",
    "📊 Métricas",
    "🕐 Controles Periódicos",
    "🔢 Runge-Kutta",
    "📈 Gráficos",
])

# ── TAB 1: Vector de Estado ───────────────────────────────────────────────────
with tab1:
    # Vector completo generado por la simulación.
    df = res["vector"]
    total = len(df)

    # Informa el rango de filas que se está mostrando.
    st.markdown(f"**Total de filas:** {total}  |  Mostrando filas **{j}** a **{min(j+i, total)-1}**")

    # Selecciona i filas a partir de j.
    subset = df.iloc[j:j+i]

    # Última fila, correspondiente al fin de simulación.
    ultima = df.iloc[[-1]]

    grupos_subset = grupos_con_clientes(subset)

    st.subheader("Iteraciones seleccionadas")
    st.markdown(render_tabla_multinivel(subset.fillna(""), grupos_subset), unsafe_allow_html=True)

    st.subheader("Última fila (fin de simulación)")
    # en última fila ocultar columnas temporales de RNDs
    cols_temp = ["RND 1 llegada","RND 2 llegada","Tiempo entre llegadas",
                 "RND tipo","RND atencion","Tiempo atencion",
                 "RND CL","Valor de A","RND1 Para Llevar","RND2 Para Llevar","Tiempo Preparacion Llevar",
                 "Tiempo de Preparacion CL","RND Salon","RND1 Permanencia","RND2 Permanencia","Tiempo Permanencia Salon"]
    grupos_ultima = [(g, [c for c in subs if c not in cols_temp]) for g, subs in grupos_con_clientes(ultima)]
    grupos_ultima = [(g, subs) for g, subs in grupos_ultima if subs]
    st.markdown(render_tabla_multinivel(ultima.fillna(""), grupos_ultima), unsafe_allow_html=True)

# ── TAB 2: Métricas ───────────────────────────────────────────────────────────
with tab2:
    # Muestra las métricas finales en tarjetas.
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
    # Muestra las tablas de controles periódicos solicitados.
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
    # Muestra las tablas RK generadas para pedidos en local.
    st.subheader("Tablas Runge-Kutta (pedidos en local)")

    if not res["rk"].empty:
        # Permite elegir qué tabla mostrar según el valor de A.
        a_vals = sorted(res["rk"]["A"].unique())
        sel = st.selectbox("Ver tabla para A =", a_vals)
        st.dataframe(res["rk"][res["rk"]["A"] == sel], use_container_width=True, height=350)
        st.caption(f"Total de pasos RK calculados: {len(res['rk'])}")
    else:
        st.info("No se calcularon tablas RK en esta simulación")

# ── TAB 5: Gráficos ───────────────────────────────────────────────────────────
with tab5:
    # Gráficos simples generados a partir del vector de estado.
    df = res["vector"]

    st.subheader("Cola caja a lo largo del tiempo")
    st.line_chart(df.set_index("Reloj (seg)")["Cola Caja"])

    st.subheader("Cola mostrador a lo largo del tiempo")
    st.line_chart(df.set_index("Reloj (seg)")["Cola Mostrador"])

    st.subheader("Ocupación salones")
    sal_df = df[["Reloj (seg)","Salon Rojo Ocupados","Salon Azul Ocupados"]]
    st.line_chart(sal_df.set_index("Reloj (seg)"))

    st.subheader("Clientes vivos en el sistema")
    st.area_chart(df.set_index("Reloj (seg)")["Clientes Vivos"])