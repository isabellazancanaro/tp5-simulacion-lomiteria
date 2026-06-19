from __future__ import annotations

# =============================================================================
# Simulación Lomitería – TP5 Grupo 22
# =============================================================================
# Este programa implementa una simulación de eventos discretos para una lomitería.
# El flujo general del cliente es:
# 1) Llega al negocio.
# 2) Hace cola y paga en caja.
# 3) Pasa al mostrador para esperar/preparar su pedido.
# 4) Si es para llevar, se prepara con la distribución configurada y se va al finalizar.
# 5) Si consume en el local, elige salón rojo o azul, permanece un tiempo y luego se va.
#
# Además, el aplicativo muestra:
# - Vector de estado.
# - Métricas finales.
# - Controles periódicos cada cierto tiempo.
# - Tablas de Runge-Kutta para pedidos consumidos en local.
# - Gráficos simples de evolución del sistema.

# Librerías matemáticas, aleatorias y de expresiones regulares.
import math
import random
import time
# re se usa para detectar y renombrar dinámicamente las columnas de clientes vivos.
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


def r2(x):
    """Redondea a 2 decimales preservando None e infinito."""
    if x is None:
        return None
    if isinstance(x, float) and math.isinf(x):
        return x
    return round(x, 2)


#Generacion VA>0 Distribucion Normal
def normal_box_muller_par(media: float, desv: float, rng: random.Random):

    while True:
        # Se redondean los RND al nacer para que el cálculo posterior
        # use los mismos valores que se muestran en el vector de estado.
        rnd1 = r2(rng.random())
        rnd2 = r2(rng.random())

        # Box-Muller no permite log(0). Si por redondeo RND1 queda 0.00,
        # se descarta ese par y se genera otro.
        if rnd1 <= 0:
            continue

        raiz = math.sqrt(-2 * math.log(rnd1))

        va1 = r2(raiz * math.cos(2 * math.pi * rnd2) * desv + media)
        va2 = r2(raiz * math.sin(2 * math.pi * rnd2) * desv + media)

        if va1 <= 0:
            va1 = None

        if va2 <= 0:
            va2 = None

        # Si las dos dieron inválidas, se genera otro par.
        if va1 is None and va2 is None:
            continue

        return rnd1, rnd2, va1, va2


def uniforme(a: float, b: float, rng: random.Random):
    """
    Genera una variable aleatoria uniforme continua entre a y b.

    Devuelve:
    - r: número aleatorio usado.
    - valor generado entre a y b.
    """
    r = r2(rng.random())
    return r, r2(a + r * (b - a))


def unif_disc(a: int, b: int, rng: random.Random):
    """
    Genera una variable aleatoria uniforme discreta entera entre a y b.

    Se usa para obtener el valor de A en los pedidos de consumo local.
    En el enunciado, A es uniforme discreto entre 2 y 5.
    """
    r = r2(rng.random())
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

    # Avanza hasta que L_sig supere 10, cortando en la misma fila donde ocurre.
    while True:
        k1 = f(t, l)
        k2 = f(t + h/2, l + h*k1/2)
        k3 = f(t + h/2, l + h*k2/2)
        k4 = f(t + h, l + h*k3)

        # Fórmula de Runge-Kutta de cuarto orden.
        l2 = l + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t2 = t + h

        # Guarda la fila actual (puede ser la que supera 10).
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

        # Corta en la misma iteración donde L superó 10, sin continuar.
        if l > 10:
            break

    # Como t = 1 equivale a 10 minutos, se convierte t a segundos multiplicando por 600.
    return r2(t * 600), rows


# ── simulación ────────────────────────────────────────────────────────────────

# Diccionario usado para cambiar nombres internos de columnas por nombres más claros
# al momento de mostrar el vector de estado. La simulación trabaja con nombres cortos
# internos, pero la interfaz muestra nombres más legibles para la entrega.
RENAME = {
    "iteracion": "Iteracion",
    "evento": "EVENTOS",
    "reloj_seg": "RELOJ (segundos)",
    "hora": "RELOJ (hh:mm:ss)",

    # Llegada cliente
    "rnd1_llg": "RND 1 Llegada",
    "rnd2_llg": "RND 2 Llegada",
    "va1_llg": "Tiempo Llegada 1",
    "va2_llg": "Tiempo Llegada 2",
    "prox_llg": "PROXIMA LLEGADA",

    # Fin atención caja + definición de tipo de pedido
    "r_caja": "RND Atencion Caja",
    "ta_caja": "Duracion Atencion En Caja",
    "fin_caja": "FIN ATENCION EN CAJA",
    "r_tipo": "RND Tipo Pedido",
    "tipo_pedido": "Tipo de Pedido del cliente",

    # Fin preparación pedido local
    "r_A": "RND A",
    "A": "A",
    "tp_prep": "TIEMPO PREPARACION PEDIDO LOCAL",

    # Fin preparación pedido para llevar
    "rnd1_llevar": "RND 1 Llevar",
    "rnd2_llevar": "RND 2 Llevar",
    "va1_llevar": "VA 1 Llevar",
    "va2_llevar": "VA 2 Llevar",

    # Próximos fines de preparación por empleado de mostrador
    "m1_fin": "Fin Prep Mostrador 1",
    "m2_fin": "Fin Prep Mostrador 2",
    "m3_fin": "Fin Prep Mostrador 3",

    # Elección de salón / fin comida
    "r_salon": "RND Salon",
    "salon_elegido": "A QUE SALON VA",
    "rnd1_sal_rojo": "RND 1 Comida Rojo",
    "rnd2_sal_rojo": "RND 2 Comida Rojo",
    "va1_sal_rojo": "tiempo de permanencia rojo 1",
    "va2_sal_rojo": "tiempo de permanencia rojo 2",
    "fin_sal_rojo": "FIN COMIDA EN SALON ROJO",
    "rnd1_sal_azul": "RND 1 Comida Azul",
    "rnd2_sal_azul": "RND 2 Comida Azul",
    "va1_sal_azul": "tiempo de permanencia azul 1",
    "va2_sal_azul": "tiempo de permanencia azul 2",
    "fin_sal_azul": "FIN COMIDA EN SALON AZUL",

    # Empleado caja
    "caja_est": "Estado Caja",
    "caja_ac": "AC Tiempo Ocupado Caja",
    "cola_caja": "COLA Clientes en Caja",
    "max_cola_caja": "MAX cantidad clientes cola en caja",

    # Empleados mostrador
    "m1_est": "M1 Estado", "m2_est": "M2 Estado", "m3_est": "M3 Estado",
    "m1_ac": "M1 AC Tiempo Ocupado", "m2_ac": "M2 AC Tiempo Ocupado", "m3_ac": "M3 AC Tiempo Ocupado",
    "cola_most": "COLA MOSTRADOR",
    "max_cola_most": "MAX cantidad clientes cola en mostrador",

    # Salones
    "r_estado":    "Rojo Estado",
    "r_ocup":      "Rojo Cantidad Personas en Salon",
    "r_esp":       "Rojo COLA",
    "r_lleno_ini": "Rojo Tiempo inicio LLENO",
    "r_ac_lleno":  "Rojo AC Tiempo Lleno",
    "a_estado":    "Azul Estado",
    "a_ocup":      "Azul Cantidad Personas en Salon",
    "a_esp":       "Azul COLA",
    "a_lleno_ini": "Azul Tiempo Inicio LLENO",
    "a_ac_lleno":  "Azul AC Tiempo Lleno",

    # Variables estadísticas
    "ac_cc": "AC Tiempo Cola en Caja",
    "n_cc": "CANTIDAD Clientes Iniciaron Atencion Caja",
    "ac_perm": "AC tiempo permanencia en local",
    "n_perm": "Cant. clientes que salieron del sistema",
    "ac_mostr_total": "AC Tiempo Ocupado Empleados",
    "vivos": "Clientes Vivos",
}

# Etiquetas usadas para renombrar las columnas dinámicas de clientes vivos.
# Ejemplo: cli3_estado pasa a mostrarse como C3 ESTADO.
CLI_FIELD_LABELS = {
    "id": "ID",
    "est": "ESTADO",
    "llg": "Tiempo Llegada",
    "tipo": "TIPO Pedido",
    "salon": "SALON",
    "salida": "Tiempo Salida",
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

    # Criterio de corte de la simulación:
    # - "tiempo": la simulación termina al llegar a hora_fin.
    # - "iteraciones": la simulación termina al alcanzar max_it eventos, sin cortar por hora_fin.
    modo_corte = p.get("modo_corte", "tiempo")
    corte_por_tiempo = modo_corte == "tiempo"
    fin = p["hora_fin"] if corte_por_tiempo else INF

    it = 0
    max_it = int(p["max_it"])

    # Rango del vector de estado que se desea conservar en memoria.
    # Se usa un límite superior exclusivo para mantener la misma lógica que df.iloc[j:j+i]:
    # si guardar_desde = 100 y guardar_hasta_excl = 150, se guardan 50 filas.
    guardar_desde = int(p.get("guardar_desde", 0))
    guardar_hasta_excl = int(p.get("guardar_hasta_excl", max_it + 1))

    # Cantidad máxima de objetos cliente que se dibujan en cada fila visible.
    # La simulación conserva todos los clientes internamente, pero el vector mostrado
    # usa una ventana acotada de clientes para evitar tablas HTML enormes cuando
    # se simulan muchas iteraciones y la cola crece demasiado.
    max_clientes_vector = int(p.get("max_clientes_vector", 30))

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

    # vector guarda solamente las filas visibles solicitadas.
    # La simulación corre completa, pero no se conserva todo el vector en memoria.
    vector: List[dict] = []

    # ultima_fila conserva siempre la última fotografía tomada, aunque no esté dentro
    # del rango visible. Esto permite mostrar la fila final sin guardar el vector completo.
    ultima_fila: Optional[dict] = None

    # rk_rows guarda todas las filas de Runge-Kutta generadas para pedidos locales.
    rk_rows: List[dict] = []
    rk_calculados: set = set()  # valores de A ya calculados, para no repetir la tabla

    # Cache de Runge-Kutta.
    # El tiempo de preparación local depende solamente de A y h_rk.
    # Antes se recalculaba la misma tabla para cada cliente local, aunque A sólo puede valer 2, 3, 4 o 5.
    # Ahora se calcula una vez por valor de A y se reutiliza el resultado.
    rk_cache: Dict[tuple, tuple] = {}

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

    # -------------------------------------------------------------------------
    # Memorias Box-Muller por evento normal
    # -------------------------------------------------------------------------
    # Cada evento normal tiene su propia memoria.
    #
    # La idea es trabajar como en el vector de estado de Excel:
    # - Se generan RND1 y RND2.
    # - Se calculan VA1 con coseno y VA2 con seno.
    # - Se usa VA1 para calcular el próximo evento.
    # - VA2 queda pendiente y se arrastra hasta ser usada.
    #
    # Hay un solo "próximo evento"; no existen próximo evento 1 y próximo evento 2.

    def nueva_memoria_bm():
        return {
            "va2_pendiente": None,
        }

    bm_llegada = nueva_memoria_bm()
    bm_llevar = nueva_memoria_bm()
    bm_permanencia: Dict[tuple, dict] = {}

    def programar_normal_bm(memoria: dict, media: float, desv: float, ahora: float):
        """
        Programa el próximo evento que usa distribución normal con Box-Muller completo.

        Lógica:
        - Si existe VA2 pendiente, se usa esa VA2 para programar el próximo evento.
        - Si no existe VA2 pendiente, se generan RND1 y RND2.
        - Con esos RND se calculan VA1 por coseno y VA2 por seno.
        - Si VA1 existe, se usa VA1 ahora y VA2 queda pendiente.
        - Si VA1 fue descartada pero VA2 existe, se usa VA2 ahora y no queda nada pendiente.

        En el vector de estado:
        - RND1, RND2 y VA1 sólo aparecen cuando se genera un nuevo par.
        - VA2 se arrastra hasta ser usada.
        """

        # Caso 1: ya había una VA2 pendiente de una generación anterior.
        if memoria["va2_pendiente"] is not None:
            tiempo = r2(memoria["va2_pendiente"])
            proximo = r2(ahora + tiempo)

            datos = {
                "rnd1": None,
                "rnd2": None,
                "va1": None,
                "va2": tiempo,
                "normal_usada": "VA2 (sen)",
            }

            # La VA2 pendiente ya fue usada, por lo tanto se limpia.
            memoria["va2_pendiente"] = None

            return tiempo, proximo, datos

        # Caso 2: no había VA2 pendiente, entonces se genera un nuevo par.
        rnd1, rnd2, va1, va2 = normal_box_muller_par(media, desv, rng)

        datos = {
            "rnd1": rnd1,
            "rnd2": rnd2,
            "va1": va1,
            "va2": va2,
            "normal_usada": None,
        }

        # Si VA1 existe, se usa primero.
        if va1 is not None:
            tiempo = r2(va1)
            datos["normal_usada"] = "VA1 (cos)"

            # Si VA2 también existe, queda pendiente para la próxima ocurrencia.
            memoria["va2_pendiente"] = va2

        # Si VA1 fue descartada, necesariamente VA2 existe.
        else:
            tiempo = r2(va2)
            datos["normal_usada"] = "VA2 (sen)"

            # Como se usó VA2 ahora, no queda nada pendiente.
            memoria["va2_pendiente"] = None

        proximo = r2(ahora + tiempo)
        return tiempo, proximo, datos
    
    def va2_pendiente_perm_salon(salon: str, ahora: float):
        """
        Devuelve la VA2 pendiente correspondiente al salón y a la franja horaria actual.

        Se usa sólo para mostrar el arrastre visual de VA2 en el vector de estado.
        No mezcla VA2 de rojo con azul ni de franjas horarias distintas.
        """
        med, dev = permanencia_params(salon, ahora)
        clave_bm_perm = (salon, med, dev)

        if clave_bm_perm not in bm_permanencia:
            return None

        return bm_permanencia[clave_bm_perm]["va2_pendiente"]


    # Contadores de tipos de pedido y salón elegido.
    n_llevar = 0
    n_local = 0
    n_rojo = 0
    n_azul = 0

    # -------------------------------------------------------------------------
    # Inicialización de próximos eventos
    # -------------------------------------------------------------------------
    # Se genera la primera llegada antes de tomar la fila inicial del vector,
    # usando Box-Muller completo.
    tel, prox_llg, bm_datos_llegada = programar_normal_bm(
        bm_llegada,
        p["media_llg"],
        p["desv_llg"],
        reloj
    )

    # Próximos controles periódicos:
    # - cola del mostrador cada ctrl_most segundos.
    # - ocupación de salones cada ctrl_sal segundos.
    prox_ctrl_m = r2(p["hora_inicio"] + p["ctrl_most"])
    prox_ctrl_s = r2(p["hora_inicio"] + p["ctrl_sal"])

    last: dict = {
        "rnd1_llg": bm_datos_llegada["rnd1"],
        "rnd2_llg": bm_datos_llegada["rnd2"],
        "va1_llg": bm_datos_llegada["va1"],
        "va2_llg": bm_datos_llegada["va2"],
    }

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
        c["ini_caja"] = r2(ahora)

        # Tiempo en cola de caja = momento en que empieza atención - momento de llegada.
        c["t_cc"] = r2(ahora - c["llg"])
        ac_cc = r2(ac_cc + c["t_cc"])
        n_cc += 1

        # Tiempo de atención en caja: uniforme entre caja_a y caja_b.
        r, ta = uniforme(p["caja_a"], p["caja_b"], rng)
        c["fin_caja"] = r2(ahora + ta)

        # Actualización del recurso caja.
        caja_est = "ocupado"
        caja_cli = cid
        caja_ini = r2(ahora)

        # Guarda los valores aleatorios para que salgan en la fila del vector de estado.
        last.update({"r_caja": r2(r), "ta_caja": r2(ta)})

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
        c["ini_prep"] = r2(ahora)

        # Tiempo en cola del mostrador = inicio de preparación - llegada al mostrador.
        c["t_cm"] = r2(ahora - c["llg_most"])
        ac_cm = r2(ac_cm + c["t_cm"])
        n_cm += 1

        if c["tipo"] == "llevar":
            # Preparación para llevar usando Box-Muller completo.
            # llevar_a actúa como media y llevar_b como desvío estándar.
            tp, fin_pl, bm_datos_llevar = programar_normal_bm(
                bm_llevar,
                p["llevar_a"],
                p["llevar_b"],
                ahora
            )

            last.update({
                "rnd1_llevar": bm_datos_llevar["rnd1"],
                "rnd2_llevar": bm_datos_llevar["rnd2"],
                "va1_llevar": bm_datos_llevar["va1"],
                "va2_llevar": bm_datos_llevar["va2"],
                "fin_prep_llevar": r2(fin_pl),
            })
        else:
            # Preparación para consumo local: se genera A y luego se calcula RK.
            ra, av = unif_disc(2, 5, rng)

            clave_rk = (av, p["h_rk"])
            if clave_rk not in rk_cache:
                rk_cache[clave_rk] = rk_local(av, p["h_rk"])

            tp, rk = rk_cache[clave_rk]

            # Solo se agrega la tabla RK si es la primera vez que aparece este valor de A.
            if av not in rk_calculados:
                rk_rows.extend(rk)
                rk_calculados.add(av)

            # Se guardan RND, A, tiempo de preparación y fin de preparación local.
            last.update({
                "r_A": r2(ra),
                "A": av,
                "tp": r2(tp),
                "fin_prep_local": r2(ahora + tp),
            })

        # Programa el fin de preparación y ocupa al empleado correspondiente.
        c["fin_prep"] = r2(ahora + tp)
        mostr[ei].update({"est": "ocupado", "cli": cid, "ini": r2(ahora), "fin": c["fin_prep"]})

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
        c["ini_salon"] = r2(ahora)

        # Obtiene media y desvío de permanencia según salón y hora actual.
        med, dev = permanencia_params(c["salon"], ahora)

        # Cada combinación de salón + media + desvío tiene su propia memoria Box-Muller.
        # Esto evita mezclar una VA2 pendiente generada con una distribución con otra distinta.
        clave_bm_perm = (c["salon"], med, dev)

        if clave_bm_perm not in bm_permanencia:
            bm_permanencia[clave_bm_perm] = nueva_memoria_bm()

        # Tiempo de permanencia en salón usando Box-Muller completo.
        tp, fin_salon, bm_datos_salon = programar_normal_bm(
            bm_permanencia[clave_bm_perm],
            med,
            dev,
            ahora
        )

        c["fin_salon"] = r2(fin_salon)

        # Registra el próximo evento de salida de salón para este cliente.
        salida_sal[cid] = c["fin_salon"]

        if c["salon"] == "rojo":
            last.update({
                "rnd1_sal_rojo": bm_datos_salon["rnd1"],
                "rnd2_sal_rojo": bm_datos_salon["rnd2"],
                "va1_sal_rojo": bm_datos_salon["va1"],
                "va2_sal_rojo": bm_datos_salon["va2"],
                "fin_sal_rojo": r2(c["fin_salon"]),
            })
        else:
            last.update({
                "rnd1_sal_azul": bm_datos_salon["rnd1"],
                "rnd2_sal_azul": bm_datos_salon["rnd2"],
                "va1_sal_azul": bm_datos_salon["va1"],
                "va2_sal_azul": bm_datos_salon["va2"],
                "fin_sal_azul": r2(c["fin_salon"]),
            })
        
        if c["salon"] == "rojo":
            salon_r.add(cid)

            # Si el salón acaba de quedar lleno, se guarda desde cuándo está lleno.
            if len(salon_r) >= p["cap_r"] and salon_r_lleno_ini is None:
                salon_r_lleno_ini = r2(ahora)
        else:
            salon_a.add(cid)

            # Si el salón acaba de quedar lleno, se guarda desde cuándo está lleno.
            if len(salon_a) >= p["cap_a"] and salon_a_lleno_ini is None:
                salon_a_lleno_ini = r2(ahora)

    def cerrar(cid, ahora):
        """
        Cierra la vida de un cliente en el sistema.

        Se usa cuando:
        - termina la preparación de un pedido para llevar,
        - o un cliente que consumió en salón se retira.
        """
        nonlocal ac_perm, n_perm, salon_r_lleno_ini, salon_a_lleno_ini, ac_r_lleno, ac_a_lleno

        # Se marca como retirado pero se mantiene en el diccionario para estabilidad de columnas.
        c = clientes[cid]
        c["est"] = "retirado"
        c["salida"] = r2(ahora)

        # Permanencia total en el negocio.
        c["perm"] = r2(ahora - c["llg"])
        ac_perm = r2(ac_perm + c["perm"])
        n_perm += 1

        # Control de tiempo acumulado con salón rojo lleno.
        # Si estaba lleno y luego de la salida queda con espacio, termina el período lleno.
        if c.get("salon") == "rojo" and salon_r_lleno_ini is not None and len(salon_r) < p["cap_r"]:
            ac_r_lleno = r2(ac_r_lleno + r2(ahora - salon_r_lleno_ini))
            salon_r_lleno_ini = None

        # Control de tiempo acumulado con salón azul lleno.
        if c.get("salon") == "azul" and salon_a_lleno_ini is not None and len(salon_a) < p["cap_a"]:
            ac_a_lleno = r2(ac_a_lleno + r2(ahora - salon_a_lleno_ini))
            salon_a_lleno_ini = None


    def ac_caja_sincronico(ahora: float) -> float:
        """
        Acumulador sincrónico de ocupación de caja.

        caja_ac guarda los períodos ya cerrados. Si la caja está ocupada,
        se suma además el tiempo transcurrido desde caja_ini hasta ahora,
        sin modificar el acumulador base para evitar doble conteo cuando
        luego ocurra el fin de atención.
        """
        if caja_est == "ocupado" and caja_ini is not None:
            return r2(caja_ac + r2(ahora - caja_ini))
        return r2(caja_ac)

    def ac_mostrador_sincronico(indice: int, ahora: float) -> float:
        """
        Acumulador sincrónico de ocupación de un empleado de mostrador.

        mostr[indice]["ac"] contiene los períodos ya finalizados. Si el
        empleado está ocupado, se suma el tramo en curso desde su inicio
        hasta el reloj actual.
        """
        empleado = mostr[indice]
        ac_base = empleado["ac"]
        if empleado["est"] == "ocupado" and empleado["ini"] is not None:
            return r2(ac_base + r2(ahora - empleado["ini"]))
        return r2(ac_base)

    def ac_mostrador_total_sincronico(ahora: float) -> float:
        """Acumulador sincrónico total de los tres empleados de mostrador."""
        return r2(sum(ac_mostrador_sincronico(i, ahora) for i in range(3)))


    def estado_caja_vector(est: str) -> str:
        """Estados de caja según el vector final del Excel."""
        return {"libre": "L", "ocupado": "OC"}.get(est, est)

    def estado_mostrador_vector(est: str) -> str:
        """Estados de mostrador según el vector final del Excel."""
        return {"libre": "L", "ocupado": "PP"}.get(est, est)

    def estado_salon_vector(cantidad: int, capacidad: int) -> str:
        """Estados de salón: Vacío, Con Lugar o Lleno."""
        if cantidad <= 0:
            return "V"
        if cantidad >= capacidad:
            return "LL"
        return "CL"

    def estado_cliente_vector(c: dict):
        """Estados del objeto temporal cliente para mostrar en el vector."""
        if c.get("est") == "retirado":
            # El cliente ya no pertenece al sistema. En la fila del evento de salida
            # se conserva el ID y el Tiempo Salida, pero no se fuerza un estado nuevo
            # que no estaba en la definición del Excel.
            return None

        return {
            "cola_caja": "EC",
            "en_caja": "SAC",
            "cola_most": "EM",
            "prep": "SAM",
            "en_salon": "C",
            "esp_r": "ES",
            "esp_a": "ES",
        }.get(c.get("est"), c.get("est"))

    def tipo_cliente_vector(c: dict):
        if c.get("tipo") == "llevar":
            return "LLEVAR"
        if c.get("tipo") == "local":
            return "LOCAL"
        return None

    def salon_cliente_vector(c: dict):
        if c.get("salon") == "rojo":
            return "ROJO"
        if c.get("salon") == "azul":
            return "AZUL"
        return None

    def tiempo_salida_cliente_vector(c: dict):
        """
        Tiempo de salida visible del cliente.
        - Para llevar: fin de preparación, porque ahí se retira.
        - Local: fin de comida en salón, cuando ya fue programado.
        """
        if c.get("tipo") == "llevar":
            return r2(c.get("fin_prep"))
        if c.get("fin_salon") is not None:
            return r2(c.get("fin_salon"))
        return None

    def nombre_evento_vector(evento: str) -> str:
        """Nombres de eventos alineados con la versión final del Excel."""
        cid = last.get("evento_cliente_id")

        def con_cliente(nombre: str) -> str:
            return f"{nombre}({cid})" if cid is not None else nombre

        if evento == "inicializacion":
            return "inicializacion"
        if evento == "llegada":
            return con_cliente("llegada_cliente")
        if evento == "fin_caja":
            return con_cliente("fin_atencion_caja")
        if evento == "fin_prep":
            c = clientes.get(cid, {})
            if c.get("tipo") == "llevar":
                return con_cliente("fin_preparacion_pedido_llevar")
            if c.get("tipo") == "local":
                return con_cliente("fin_preparacion_pedido_local")
            return con_cliente("fin_preparacion")
        if evento == "salida_salon":
            c = clientes.get(cid, {})
            if c.get("salon") == "azul":
                return con_cliente("fin_comida_azul")
            return con_cliente("fin_comida_rojo")
        if evento == "ctrl_most":
            return "control_cola_mostrador"
        if evento == "ctrl_sal":
            return "control_salones"
        if evento == "fin_sim":
            return "fin_sim"
        return evento

    # -------------------------------------------------------------------------
    # Slots estables para el objeto temporal Cliente
    # -------------------------------------------------------------------------
    # Antes los clientes visibles se elegían de nuevo en cada fila y se dibujaban
    # en C1, C2, C3, ... según prioridad de esa fila. Eso hacía que un mismo
    # cliente se moviera de columna, o que una misma columna mostrara IDs distintos.
    #
    # Ahora, dentro del rango visible solicitado, cada ID real queda asociado a un
    # slot fijo. Si el cliente deja el sistema, su slot queda vacío en las filas
    # posteriores y no se reutiliza para otro cliente dentro del mismo vector visible.
    cliente_slot_visible: Dict[int, int] = {}
    slot_cliente_visible: Dict[int, int] = {}

    def candidatos_clientes_para_vector(clientes_activos: list) -> list:
        """
        Devuelve IDs candidatos a ocupar slots de cliente en el vector visible.

        La selección sigue siendo sólo visual: no modifica la lógica de simulación.
        Se prioriza el cliente del evento, recursos ocupados, colas, salones y luego
        clientes activos recientes. El cliente del evento se permite aunque se haya
        retirado en esta misma fila, para que pueda verse su Tiempo Salida.
        """
        if max_clientes_vector <= 0:
            return []

        seleccionados = []
        vistos = set()
        cid_evento = last.get("evento_cliente_id")

        def agregar(cid, permitir_retirado: bool = False):
            if cid is None or cid in vistos:
                return
            c = clientes.get(cid)
            if c is None:
                return
            if c.get("est") == "retirado" and not permitir_retirado:
                return
            vistos.add(cid)
            seleccionados.append(cid)

        # 1) Cliente que disparó el evento actual.
        agregar(cid_evento, permitir_retirado=True)

        # 2) Cliente en caja y clientes atendidos/preparados por mostrador.
        agregar(caja_cli)
        for empleado in mostr:
            agregar(empleado.get("cli"))

        # 3) Primeros clientes de las colas.
        for q in (cola_caja, cola_most, esp_r, esp_a):
            for cid in list(q):
                agregar(cid)
                if len(seleccionados) >= max_clientes_vector:
                    break
            if len(seleccionados) >= max_clientes_vector:
                break

        # 4) Clientes en salón, priorizando los que salen antes.
        if len(seleccionados) < max_clientes_vector:
            en_salon = sorted(
                list(salon_r | salon_a),
                key=lambda cid: salida_sal.get(cid, INF)
            )
            for cid in en_salon:
                agregar(cid)
                if len(seleccionados) >= max_clientes_vector:
                    break

        # 5) Completa con clientes activos más recientes.
        if len(seleccionados) < max_clientes_vector:
            for c in sorted(clientes_activos, key=lambda cv: cv["id"], reverse=True):
                agregar(c["id"])
                if len(seleccionados) >= max_clientes_vector:
                    break

        return seleccionados

    def asignar_slots_clientes(candidatos: list):
        """Asigna slots fijos C1, C2, ... a IDs reales, sin reutilizarlos."""
        if max_clientes_vector <= 0:
            return

        for cid in candidatos:
            if cid in cliente_slot_visible:
                continue
            if len(slot_cliente_visible) >= max_clientes_vector:
                break

            for slot in range(1, max_clientes_vector + 1):
                if slot not in slot_cliente_visible:
                    cliente_slot_visible[cid] = slot
                    slot_cliente_visible[slot] = cid
                    break

    def snap(evento, force: bool = False):
        """
        Toma una fotografía del sistema en el instante actual.

        Para mejorar rendimiento, la fila completa del vector sólo se construye
        cuando realmente se va a conservar: rango visible, fila final o force=True.
        Esto evita armar miles de diccionarios grandes que después no se muestran.
        """
        debe_guardar = force or (guardar_desde <= it < guardar_hasta_excl) or evento == "fin_sim"
        if not debe_guardar:
            return

        clientes_activos = [cv for cv in clientes.values() if cv["est"] != "retirado"]

        row = {
            # Datos generales de la fila.
            "iteracion": it,
            "evento": nombre_evento_vector(evento),
            "reloj_seg": r2(reloj),
            "hora": fmt(reloj),

            # Llegadas: Box-Muller completo.
            # RND1, RND2 y VA1 sólo se muestran cuando se genera un nuevo par.
            # VA2 se arrastra visualmente mientras esté pendiente.
            "rnd1_llg": last.get("rnd1_llg"),
            "rnd2_llg": last.get("rnd2_llg"),
            "va1_llg": last.get("va1_llg"),
            "va2_llg": (
                last.get("va2_llg")
                if last.get("va2_llg") is not None
                else bm_llegada["va2_pendiente"]
            ),
            "prox_llg": r2(prox_llg) if prox_llg < INF else None,

            # Tipo de pedido: para llevar o local.
            "r_tipo": last.get("r_tipo"),
            "tipo_pedido": last.get("tipo_pedido"),

            # Caja: RND, tiempo de atención y próximo fin de atención.
            "r_caja": last.get("r_caja"),
            "ta_caja": last.get("ta_caja"),
            "fin_caja": r2(clientes[caja_cli]["fin_caja"]) if caja_cli in clientes else None,
            # Preparación local mediante RK y preparación para llevar.
            "r_A": last.get("r_A"), 
            "A": last.get("A"),

            # Preparación para llevar: Box-Muller completo.
            # RND1, RND2 y VA1 sólo se muestran cuando se genera un nuevo par.
            # VA2 se arrastra visualmente mientras esté pendiente.
            "rnd1_llevar": last.get("rnd1_llevar"),
            "rnd2_llevar": last.get("rnd2_llevar"),
            "va1_llevar": last.get("va1_llevar"),
            "va2_llevar": (
                last.get("va2_llevar")
                if last.get("va2_llevar") is not None
                else bm_llevar["va2_pendiente"]
            ),
            "fin_prep_llevar": last.get("fin_prep_llevar"),

            "tp_prep": last.get("tp"),
            "fin_prep_local": last.get("fin_prep_local"),
            "salon_elegido": last.get("salon_elegido"),

            # Elección de salón.
            "r_salon": last.get("r_salon"),

            # Permanencia en salón rojo: Box-Muller completo.
            "rnd1_sal_rojo": last.get("rnd1_sal_rojo"),
            "rnd2_sal_rojo": last.get("rnd2_sal_rojo"),
            "va1_sal_rojo": last.get("va1_sal_rojo"),
            "va2_sal_rojo": (
                last.get("va2_sal_rojo")
                if last.get("va2_sal_rojo") is not None
                else va2_pendiente_perm_salon("rojo", reloj)
            ),
            "fin_sal_rojo": last.get("fin_sal_rojo"),

            # Permanencia en salón azul: Box-Muller completo.
            "rnd1_sal_azul": last.get("rnd1_sal_azul"),
            "rnd2_sal_azul": last.get("rnd2_sal_azul"),
            "va1_sal_azul": last.get("va1_sal_azul"),
            "va2_sal_azul": (
                last.get("va2_sal_azul")
                if last.get("va2_sal_azul") is not None
                else va2_pendiente_perm_salon("azul", reloj)
            ),
            "fin_sal_azul": last.get("fin_sal_azul"),
            
            # Estado de la caja, cola de caja y acumulador de ocupación.
            "caja_est": estado_caja_vector(caja_est), "caja_cli": caja_cli,
            "cola_caja": len(cola_caja), "max_cola_caja": max_cc,
            "caja_ac": ac_caja_sincronico(reloj),
            
            # Próximos fines de preparación por empleado.
            **{f"m{i+1}_fin":   None if mostr[i]["fin"] == INF else r2(mostr[i]["fin"]) for i in range(3)},

            # Estado de los tres empleados del mostrador.
            **{f"m{i+1}_est":   estado_mostrador_vector(mostr[i]["est"]) for i in range(3)},
            **{f"m{i+1}_cli":   mostr[i]["cli"]               for i in range(3)},
            **{f"m{i+1}_ac":    ac_mostrador_sincronico(i, reloj) for i in range(3)},
            "cola_most": len(cola_most), "max_cola_most": max_cm,
            # Ocupación y esperas de los salones.
            "r_estado": estado_salon_vector(len(salon_r), p["cap_r"]),
            "a_estado": estado_salon_vector(len(salon_a), p["cap_a"]),
            "r_ocup": len(salon_r), "a_ocup": len(salon_a),
            "r_esp": len(esp_r),    "a_esp": len(esp_a),
            "r_lleno_ini": r2(salon_r_lleno_ini) if salon_r_lleno_ini is not None else None,
            "a_lleno_ini": r2(salon_a_lleno_ini) if salon_a_lleno_ini is not None else None,
            "r_ac_lleno": r2(ac_r_lleno), "a_ac_lleno": r2(ac_a_lleno),
            # Acumuladores y contadores usados para calcular métricas finales.
            "ac_perm": r2(ac_perm),  "n_perm":  n_perm,
            "ac_cc":   r2(ac_cc),    "n_cc":    n_cc,
            "ac_mostr_total": ac_mostrador_total_sincronico(reloj),
            "vivos": len(clientes_activos),
        }

        # Clientes visibles en el vector de estado.
        # Cada ID real mantiene siempre el mismo slot visual dentro del rango mostrado.
        candidatos = candidatos_clientes_para_vector(clientes_activos)
        asignar_slots_clientes(candidatos)

        cid_evento = last.get("evento_cliente_id")
        for slot in range(1, max_clientes_vector + 1):
            cid = slot_cliente_visible.get(slot)
            if cid is None:
                continue

            cv = clientes.get(cid)
            if cv is None:
                continue

            # Si el cliente ya salió, sólo se lo conserva en la fila exacta del
            # evento que lo hizo salir. En filas posteriores el slot queda vacío,
            # pero no se reutiliza para otro cliente.
            if cv.get("est") == "retirado" and cid != cid_evento:
                continue

            row[f"cli{slot}_id"]     = cv["id"]
            row[f"cli{slot}_est"]    = estado_cliente_vector(cv)
            row[f"cli{slot}_llg"]    = r2(cv["llg"])
            row[f"cli{slot}_tipo"]   = tipo_cliente_vector(cv)
            row[f"cli{slot}_salon"]  = salon_cliente_vector(cv)
            row[f"cli{slot}_salida"] = tiempo_salida_cliente_vector(cv)

        nonlocal ultima_fila
        ultima_fila = row
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
    while it < max_it:
        # Limpia los datos aleatorios del evento anterior.
        # Sólo se cargarán en last los RNDs usados en el evento actual.
        last = {}

        # Próximo fin de caja, si hay cliente en caja.
        fc = clientes[caja_cli]["fin_caja"] if caja_cli in clientes else INF

        # Próximo fin de preparación entre los 3 empleados del mostrador.
        fp = min(m["fin"] for m in mostr)

        # Próxima salida de salón, si hay clientes en algún salón.
        fs = min(salida_sal.values()) if salida_sal else INF

        # Selección manual del próximo evento.
        # Evita construir un diccionario y recorrerlo en cada iteración.
        # Se conserva la prioridad anterior ante empates: llegada, fin_caja,
        # fin_prep, salida_salon, ctrl_most, ctrl_sal y fin_sim.
        ev = "llegada"
        t_ev = prox_llg

        if fc < t_ev:
            ev, t_ev = "fin_caja", fc
        if fp < t_ev:
            ev, t_ev = "fin_prep", fp
        if fs < t_ev:
            ev, t_ev = "salida_salon", fs
        if prox_ctrl_m < t_ev:
            ev, t_ev = "ctrl_most", prox_ctrl_m
        if prox_ctrl_s < t_ev:
            ev, t_ev = "ctrl_sal", prox_ctrl_s

        # En modo por tiempo, fin_sim corta si es el próximo instante o si empata
        # con otro evento. Esto mantiene la lógica previa del chequeo reloj >= fin.
        if corte_por_tiempo and fin <= t_ev:
            ev, t_ev = "fin_sim", fin

        reloj = r2(t_ev)
        it += 1

        if ev == "llegada":
            # -----------------------------------------------------------------
            # Evento: llegada de cliente
            # -----------------------------------------------------------------
            # Se crea un nuevo cliente en estado cola_caja.
            c = {
                "id": nxt_id,
                "llg": r2(reloj),
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

            last["evento_cliente_id"] = nxt_id
            nxt_id += 1

            # Se programa la próxima llegada usando Box-Muller completo.
            tel, prox_llg, bm_datos_llegada = programar_normal_bm(
                bm_llegada,
                p["media_llg"],
                p["desv_llg"],
                reloj
            )

            last.update({
                "rnd1_llg": bm_datos_llegada["rnd1"],
                "rnd2_llg": bm_datos_llegada["rnd2"],
                "va1_llg": bm_datos_llegada["va1"],
                "va2_llg": bm_datos_llegada["va2"],
            })

        elif ev == "fin_caja":
            # -----------------------------------------------------------------
            # Evento: fin de atención en caja
            # -----------------------------------------------------------------
            # El cliente que estaba en caja pasa al mostrador.
            nonlocal_cid = caja_cli
            last["evento_cliente_id"] = nonlocal_cid
            c = clientes[nonlocal_cid]
            c["est"] = "cola_most"
            c["llg_most"] = r2(reloj)

            # Se determina si compra para llevar o para consumir en local.
            # En este momento sólo se define el tipo de pedido.
            # Si es local, el salón todavía no se elige acá:
            # se elegirá recién cuando termine la preparación en mostrador.
            rt = r2(rng.random())
            if rt < p["prob_llevar"]:
                c["tipo"] = "llevar"
                n_llevar += 1

                last.update({
                    "r_tipo": r2(rt),
                    "tipo_pedido": "LLEVAR",
                })
            else:
                c["tipo"] = "local"
                n_local += 1

                last.update({
                    "r_tipo": r2(rt),
                    "tipo_pedido": "LOCAL",
                })

            # Se acumula el tiempo ocupado de caja para calcular ocupación.
            if caja_ini is not None:
                caja_ac = r2(caja_ac + r2(reloj - caja_ini))

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
            last["evento_cliente_id"] = cid

            # Se acumula el tiempo ocupado de ese empleado.
            if mostr[ei]["ini"] is not None:
                mostr[ei]["ac"] = r2(mostr[ei]["ac"] + r2(reloj - mostr[ei]["ini"]))

            # El empleado queda libre.
            mostr[ei].update({"est": "libre", "cli": None, "ini": None, "fin": INF})

            c = clientes[cid]

            # Si el pedido era para llevar, el cliente se va.
            # Si era local, intenta ingresar al salón elegido.
            if c["tipo"] == "llevar":
                cerrar(cid, reloj)
            else:
                rs = r2(rng.random())

                c["salon"] = "rojo" if rs < p["prob_rojo"] else "azul"

                if c["salon"] == "rojo":
                    n_rojo += 1
                else:
                    n_azul += 1

                last.update({
                    "r_salon": r2(rs),
                    "salon_elegido": c["salon"].upper(),
                })

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
            last["evento_cliente_id"] = cid
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
            prox_ctrl_m = r2(prox_ctrl_m + p["ctrl_most"])

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
            prox_ctrl_s = r2(prox_ctrl_s + p["ctrl_sal"])

        # Después de ejecutar el evento, se guarda la fila del vector de estado.
        snap(ev)

        # Si se llegó al fin de simulación, termina el bucle.
        if ev == "fin_sim":
            break

    # Si la simulación terminó por cantidad de iteraciones o por límite de seguridad,
    # puede no existir una fila final explícita. Se agrega una fotografía final liviana
    # con el estado alcanzado para conservar la vista de cierre de simulación.
    if ultima_fila is None or ultima_fila.get("iteracion") != it or not str(ultima_fila.get("evento", "")).startswith("fin_sim"):
        last = {}
        snap("fin_sim", force=True)

    # -------------------------------------------------------------------------
    # Cálculo final de métricas
    # -------------------------------------------------------------------------
    # La duración usada para las ocupaciones debe ser la duración realmente simulada.
    # En modo por tiempo normalmente coincide con hora_fin - hora_inicio.
    # En modo por iteraciones coincide con el reloj alcanzado al cortar por max_it.
    dur = r2(reloj - p["hora_inicio"])
    if dur <= 0:
        dur = 0.01

    # Tiempo ocupado sincrónico al instante final.
    # Incluye tramos cerrados y, si un recurso sigue ocupado al corte,
    # suma el tramo en curso hasta el reloj final.
    caja_ac_final = ac_caja_sincronico(reloj)
    ac_m = ac_mostrador_total_sincronico(reloj)

    metricas = {
        "Prom. permanencia negocio (seg)": r2(ac_perm / n_perm) if n_perm else 0,
        "Prom. tiempo cola caja (seg)": r2(ac_cc / n_cc) if n_cc else 0,
        "Prom. tiempo cola mostrador (seg)": r2(ac_cm / n_cm) if n_cm else 0,
        "% Ocupación caja": r2(caja_ac_final / dur * 100),
        "% Ocupación empleados mostrador": r2(ac_m / (3 * dur) * 100),
        "Máx. cola caja": max_cc,
        "Máx. cola mostrador": max_cm,
        "AC tiempo salón rojo lleno (seg)": r2(ac_r_lleno),
        "AC tiempo salón azul lleno (seg)": r2(ac_a_lleno),
        "Clientes finalizados": n_perm,
        "Para llevar": n_llevar,
        "En local": n_local,
        "Salón rojo": n_rojo,
        "Salón azul": n_azul,
    }

    # Se convierte el vector visible a DataFrame y se renombran columnas para que sean más claras.
    df_vector = pd.DataFrame(vector).rename(columns=RENAME)
    cli_rename = {}
    for col in df_vector.columns:
        m = re.match(r"cli(\d+)_(id|est|llg|tipo|salon|salida)", col)
        if m:
            k, campo = m.groups()
            cli_rename[col] = f"C{k} {CLI_FIELD_LABELS[campo]}"
    df_vector = df_vector.rename(columns=cli_rename)

    df_ultima = pd.DataFrame([ultima_fila]).rename(columns=RENAME) if ultima_fila else pd.DataFrame()
    if not df_ultima.empty:
        cli_rename_ultima = {}
        for col in df_ultima.columns:
            m = re.match(r"cli(\d+)_(id|est|llg|tipo|salon|salida)", col)
            if m:
                k, campo = m.groups()
                cli_rename_ultima[col] = f"C{k} {CLI_FIELD_LABELS[campo]}"
        df_ultima = df_ultima.rename(columns=cli_rename_ultima)

    return {
        "vector": df_vector,
        "ultima_fila": df_ultima,
        "total_iteraciones": it,
        "modo_corte": modo_corte,
        "reloj_final": r2(reloj),
        "hora_final": fmt(reloj),
        "duracion_simulada_seg": dur,
        "ctrl_most": pd.DataFrame(ctrl_most) if ctrl_most else pd.DataFrame(),
        "ctrl_sal":  pd.DataFrame(ctrl_sal)  if ctrl_sal  else pd.DataFrame(),
        "rk":        pd.DataFrame(rk_rows)   if rk_rows   else pd.DataFrame(),
        "metricas":  metricas,
    }

# ── tabla multinivel ───────────────────────────────────────────────────────────
# Estas estructuras permiten mostrar el vector de estado con encabezados agrupados.
# La simulación sigue generando un DataFrame normal, pero esta sección lo renderiza
# como una tabla HTML con grupos como "Proxima Llegada", "Empleado Caja", etc.

GRUPOS = [
    ("", ["Iteracion", "EVENTOS", "RELOJ (segundos)", "RELOJ (hh:mm:ss)"]),

    ("LLEGADA CLIENTE", [
        "RND 1 Llegada", "RND 2 Llegada", "Tiempo Llegada 1", "Tiempo Llegada 2", "PROXIMA LLEGADA"
    ]),

    ("FIN ATENCION CAJA", [
        "RND Atencion Caja", "Duracion Atencion En Caja", "FIN ATENCION EN CAJA",
        "RND Tipo Pedido", "Tipo de Pedido del cliente"
    ]),

    ("FIN PREPARACION PEDIDO LOCAL", [
        "RND A", "A", "TIEMPO PREPARACION PEDIDO LOCAL"
    ]),

    ("FIN PREPARACION PEDIDO PARA LLEVAR", [
        "RND 1 Llevar", "RND 2 Llevar", "VA 1 Llevar", "VA 2 Llevar"
    ]),

    ("FIN PREPARACION", [
        "Fin Prep Mostrador 1", "Fin Prep Mostrador 2", "Fin Prep Mostrador 3"
    ]),

    ("FIN COMIDA", [
        "RND Salon", "A QUE SALON VA"
    ]),

    ("FIN COMIDA ROJO", [
        "RND 1 Comida Rojo", "RND 2 Comida Rojo",
        "tiempo de permanencia rojo 1", "tiempo de permanencia rojo 2", "FIN COMIDA EN SALON ROJO"
    ]),

    ("FIN COMIDA AZUL", [
        "RND 1 Comida Azul", "RND 2 Comida Azul",
        "tiempo de permanencia azul 1", "tiempo de permanencia azul 2", "FIN COMIDA EN SALON AZUL"
    ]),

    ("empleado CAJA", [
        "Estado Caja", "AC Tiempo Ocupado Caja",
        "COLA Clientes en Caja", "MAX cantidad clientes cola en caja"
    ]),

    ("Mostrador 1", ["M1 Estado", "M1 AC Tiempo Ocupado"]),
    ("Mostrador 2", ["M2 Estado", "M2 AC Tiempo Ocupado"]),
    ("Mostrador 3", ["M3 Estado", "M3 AC Tiempo Ocupado"]),

    ("empleados MOSTRADOR", ["COLA MOSTRADOR", "MAX cantidad clientes cola en mostrador"]),
    ("ROJO", ["Rojo Estado", "Rojo Cantidad Personas en Salon", "Rojo COLA", "Rojo Tiempo inicio LLENO", "Rojo AC Tiempo Lleno"]),
    ("AZUL", ["Azul Estado", "Azul Cantidad Personas en Salon", "Azul COLA", "Azul Tiempo Inicio LLENO", "Azul AC Tiempo Lleno"]),
    ("VARIABLES ESTADISTICAS", [
        "AC Tiempo Cola en Caja", "CANTIDAD Clientes Iniciaron Atencion Caja",
        "AC tiempo permanencia en local", "Cant. clientes que salieron del sistema",
        "AC Tiempo Ocupado Empleados"
    ]),
]

# Etiquetas visuales para que el encabezado se parezca al Excel aunque internamente
# las columnas tengan nombres únicos.
DISPLAY_COL_LABELS = {
    "RND 1 Llegada": "RND 1",
    "RND 2 Llegada": "RND 2",
    "RND Atencion Caja": "RND",
    "RND Tipo Pedido": "RND",
    "RND 1 Llevar": "RND 1",
    "RND 2 Llevar": "RND 2",
    "VA 1 Llevar": "VA 1",
    "VA 2 Llevar": "VA 2",
    "Fin Prep Mostrador 1": "1",
    "Fin Prep Mostrador 2": "2",
    "Fin Prep Mostrador 3": "3",
    "RND Salon": "RND",
    "RND 1 Comida Rojo": "RND 1",
    "RND 2 Comida Rojo": "RND 2",
    "tiempo de permanencia rojo 1": "tiempo de permanencia 1",
    "tiempo de permanencia rojo 2": "tiempo de permanencia 2",
    "RND 1 Comida Azul": "RND 1",
    "RND 2 Comida Azul": "RND 2",
    "tiempo de permanencia azul 1": "tiempo de permanencia 1",
    "tiempo de permanencia azul 2": "tiempo de permanencia 2",
    "Estado Caja": "ESTADO",
    "AC Tiempo Ocupado Caja": "AC Tiempo Ocupado",
    "M1 Estado": "ESTADO",
    "M1 AC Tiempo Ocupado": "AC Tiempo Ocupado",
    "M2 Estado": "ESTADO",
    "M2 AC Tiempo Ocupado": "AC Tiempo Ocupado",
    "M3 Estado": "ESTADO",
    "M3 AC Tiempo Ocupado": "AC Tiempo Ocupado",
    "Rojo Estado": "ESTADO",
    "Rojo Cantidad Personas en Salon": "Cantidad Personas en Salon",
    "Rojo COLA": "COLA",
    "Rojo Tiempo inicio LLENO": "Tiempo inicio LLENO",
    "Rojo AC Tiempo Lleno": "AC Tiempo Lleno",
    "Azul Estado": "ESTADO",
    "Azul Cantidad Personas en Salon": "Cantidad Personas en Salon",
    "Azul COLA": "COLA",
    "Azul Tiempo Inicio LLENO": "Tiempo Inicio LLENO",
    "Azul AC Tiempo Lleno": "AC Tiempo Lleno",
    "MAX cantidad clientes cola en mostrador": "MAX cantidad clientes cola en mostrador",
}

GRUPO_COLORES = {
    # Paleta similar al Excel, pero con tonos más oscuros/saturados para que
    # el texto blanco se lea mejor en el tema oscuro de Streamlit.
    "": "#2d2d2d",
    "LLEGADA CLIENTE": "#3f7f2f",
    "FIN ATENCION CAJA": "#4f86b8",
    "FIN PREPARACION PEDIDO LOCAL": "#5f9b4c",
    "FIN PREPARACION PEDIDO PARA LLEVAR": "#4f86b8",
    "FIN PREPARACION": "#3f7f2f",
    "FIN COMIDA": "#d87918",
    "FIN COMIDA ROJO": "#c94c4c",
    "FIN COMIDA AZUL": "#2f6fae",
    "empleado CAJA": "#3f7f2f",
    "Mostrador 1": "#4f86b8",
    "Mostrador 2": "#4f86b8",
    "Mostrador 3": "#4f86b8",
    "empleados MOSTRADOR": "#4f86b8",
    "ROJO": "#c94c4c",
    "AZUL": "#2f6fae",
    "VARIABLES ESTADISTICAS": "#4f86b8",
    "CLIENTES": "#3f7f2f",
}

COLS_PROXIMO_EVENTO = {
    "PROXIMA LLEGADA",
    "FIN ATENCION EN CAJA",
    "Fin Prep Mostrador 1",
    "Fin Prep Mostrador 2",
    "Fin Prep Mostrador 3",
    "FIN COMIDA EN SALON ROJO",
    "FIN COMIDA EN SALON AZUL",
}

def _intensificar(hex_color: str) -> str:
    """Color para columnas de próximo evento.

    Mantiene el amarillo/naranja del Excel, pero con texto oscuro en el render
    para que se lea bien.
    """
    tabla = {
        "#3f7f2f": "#f2df00",
        "#4f86b8": "#f2df00",
        "#5f9b4c": "#f2df00",
        "#d87918": "#f08a1a",
        "#c94c4c": "#f2df00",
        "#2f6fae": "#f2df00",
        "#2d2d2d": "#707070",
    }
    return tabla.get(hex_color, "#f2df00")


def _color_grupo(grupo: str) -> str:
    """Devuelve color de encabezado, incluyendo grupos dinámicos de clientes."""
    if grupo.startswith("Cliente "):
        return GRUPO_COLORES["CLIENTES"]
    return GRUPO_COLORES.get(grupo, "#2d2d2d")


def grupos_con_clientes(df: pd.DataFrame) -> list:
    """
    Agrega grupos de columnas para los clientes que aparezcan en el DataFrame.

    El vector usa slots C1, C2, ... para mantener estable el ancho de la tabla.
    Dentro de cada slot se muestra el ID real del cliente.
    """
    cli_nums = sorted({int(m.group(1)) for c in df.columns if (m := re.match(r"C(\d+) ID", c))})
    grupos = list(GRUPOS)
    for k in cli_nums:
        grupos.append((f"Cliente {k}", [
            f"C{k} ID", f"C{k} ESTADO", f"C{k} Tiempo Llegada",
            f"C{k} TIPO Pedido", f"C{k} SALON", f"C{k} Tiempo Salida"
        ]))
    return grupos


def render_tabla_multinivel(df: pd.DataFrame, grupos: list) -> str:
    cols_ordenadas = []
    for grupo, subcols in grupos:
        for sc in subcols:
            if sc in df.columns and sc != "_col_ganadora":
                cols_ordenadas.append((grupo, sc))

    STICKY_WIDTHS = {"Iteracion": 70, "EVENTOS": 160, "RELOJ (segundos)": 110, "RELOJ (hh:mm:ss)": 100}
    sticky_cols = set(STICKY_WIDTHS.keys())
    sticky_offsets = {}
    acum = 0
    for sc, w in STICKY_WIDTHS.items():
        sticky_offsets[sc] = acum
        acum += w

    header1 = ""
    header2 = ""
    col_index = 0
    for grupo, subcols in grupos:
        presentes = [c for c in subcols if c in df.columns and c != "_col_ganadora"]
        if not presentes:
            continue

        color_grupo = _color_grupo(grupo)
        primer_col = presentes[0]
        es_sticky_grupo = primer_col in sticky_cols
        z_header = "z-index:4" if es_sticky_grupo else "z-index:1"
        left_grupo = f"left:{sticky_offsets.get(primer_col, 0)}px;" if es_sticky_grupo else ""

        header1 += (
            f'<th colspan="{len(presentes)}" '
            f'style="text-align:center;border:1px solid #555;padding:5px;'
            f'background:{color_grupo};color:#ffffff;font-size:0.8em;font-weight:bold;'
            f'position:sticky;top:0;{left_grupo}{z_header}">'
            f'{grupo}</th>'
        )

        for sc in presentes:
            es_proximo = sc in COLS_PROXIMO_EVENTO
            bg_col = _intensificar(color_grupo) if es_proximo else color_grupo
            es_sticky = sc in sticky_cols
            w = STICKY_WIDTHS.get(sc, 82)
            offset = sticky_offsets.get(sc, 0)
            sticky = (
                f"position:sticky;left:{offset}px;z-index:3;min-width:{w}px;max-width:{w}px;"
                if es_sticky else f"min-width:82px;"
            )
            col_index += 1

            color_texto_header = "#111111" if es_proximo else "#ffffff"
            peso_header = "bold" if es_proximo else "normal"

            header2 += (
                f'<th style="text-align:center;border:1px solid #555;padding:4px;'
                f'white-space:pre-wrap;font-size:0.72em;'
                f'background:{bg_col};color:{color_texto_header};'
                f'font-weight:{peso_header};'
                f'position:sticky;top:28px;{sticky}">'
                f'{DISPLAY_COL_LABELS.get(sc, sc).replace(" ", "<br>").replace(chr(10), "<br>")}</th>'
            )

    rows = ""
    col_idx_map = {sc: i for i, (_, sc) in enumerate(cols_ordenadas)}
    df_reset = df.reset_index(drop=True)
    for idx, row in df_reset.iterrows():
        reloj_sig = df_reset.iloc[idx + 1]["RELOJ (segundos)"] if idx + 1 < len(df_reset) else None
        rows += "<tr>"
        for grupo, subcols in grupos:
            for sc in subcols:
                if sc in df.columns and sc != "_col_ganadora":
                    val = row.get(sc, "")
                    es_sticky = sc in sticky_cols
                    w = STICKY_WIDTHS.get(sc, 82)
                    offset = sticky_offsets.get(sc, 0)
                    sticky = (
                        f"position:sticky;left:{offset}px;z-index:1;background:#1a1a1a;min-width:{w}px;max-width:{w}px;"
                        if es_sticky else "min-width:82px;"
                    )
                    es_ganadora = (
                        sc in COLS_PROXIMO_EVENTO
                        and reloj_sig is not None
                        and val != ""
                        and abs(float(val) - float(reloj_sig)) < 0.01
                    )
                    color_texto = "color:#ff4444;font-weight:bold;" if es_ganadora else ""
                    rows += (
                        f'<td style="text-align:center;border:1px solid #333;padding:3px;'
                        f'font-size:0.8em;{sticky}{color_texto}">{val}</td>'
                    )
        rows += "</tr>"

    html = f"""
    <div style="overflow-x:auto;max-height:600px;overflow-y:auto">
    <table style="border-collapse:collapse;font-family:monospace;table-layout:fixed">
        <thead style="position:sticky;top:0;z-index:5">
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
st.title("Simulación Lomitería – TP5 Grupo 22")

# Panel lateral donde el usuario carga los parámetros de la simulación.
with st.sidebar:
    st.header("Parámetros")

    # Semilla usada para reproducir la misma secuencia de números aleatorios.
    semilla = st.number_input("Semilla aleatoria", value=22, step=1)

    st.subheader("Corte de simulación")
    # Permite elegir si la simulación termina por hora fin o por cantidad exacta de eventos.
    modo_corte_ui = st.radio(
        "Finalizar simulación por",
        ["Tiempo de simulación", "Cantidad de iteraciones"],
        index=0,
    )
    modo_corte = "tiempo" if modo_corte_ui == "Tiempo de simulación" else "iteraciones"

    st.subheader("Tiempo simulación")
    # La hora de inicio se usa en ambos modos. La hora fin solo corta la corrida en modo tiempo.
    h_ini = st.number_input("Hora inicio", 0, 23, 11)
    h_fin = st.number_input(
        "Hora fin",
        0,
        24,
        15,
        disabled=(modo_corte == "iteraciones"),
        help="Solo se usa cuando el criterio de corte es por tiempo.",
    )

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
    # En esta versión se interpreta la preparación para llevar como normal positiva.
    # llevar_a = media; llevar_b = desvío.
    llevar_a = st.number_input("Media llevar (seg)", value=120.0, min_value=0.0)
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
    # En modo por iteraciones, este valor es la cantidad de eventos a simular.
    # En modo por tiempo, funciona como límite de seguridad para evitar corridas infinitas.
    if modo_corte == "iteraciones":
        max_it = st.number_input(
            "Cantidad de iteraciones",
            value=100000,
            min_value=10,
            max_value=100000,
            step=1000,
        )
    else:
        max_it = st.number_input(
            "Límite máximo de iteraciones",
            value=100000,
            min_value=10,
            max_value=100000,
            step=1000,
            help="Corte de seguridad. Normalmente la simulación termina antes, al llegar a la hora fin.",
        )

    st.subheader("Filtro vector estado")
    # Permite mostrar i filas del vector a partir de una iteración j.
    j_it = st.number_input("Desde iteración j", value=0, min_value=0, step=1)
    i_it = st.number_input("Mostrar i iteraciones", value=50, min_value=1, step=10)
    max_clientes_vector = st.number_input(
        "Máx. clientes a mostrar en el rango",
        value=30,
        min_value=0,
        max_value=200,
        step=5,
        help=(
            "Cada cliente conserva su columna dentro del rango visible. "
            "No cambia la lógica ni las métricas de la simulación."
        ),
    )

    # Botón que ejecuta la simulación.
    correr = st.button("▶ Simular", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Validaciones básicas de parámetros
# -----------------------------------------------------------------------------
errores = []
if modo_corte == "tiempo" and h_fin <= h_ini:
    errores.append("Hora fin debe ser mayor que hora inicio.")
if not (10 <= int(max_it) <= 100000):
    errores.append("La cantidad/límite de iteraciones debe estar entre 10 y 100.000.")
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
        modo_corte=modo_corte,
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
        guardar_desde=int(j_it),
        guardar_hasta_excl=int(j_it) + int(i_it),
        max_clientes_vector=int(max_clientes_vector),
    )

    # Muestra un spinner mientras corre la simulación.
    with st.spinner("Simulando..."):
        t0 = time.perf_counter()
        res = simular(params)
        res["tiempo_ejecucion_seg"] = r2(time.perf_counter() - t0)

    # Guarda los resultados en session_state para que no se pierdan al cambiar pestañas.
    st.session_state["res"] = res
    st.session_state["j"] = int(j_it)
    st.session_state["i"] = int(i_it)

    criterio_txt = "por tiempo" if res.get("modo_corte") == "tiempo" else "por cantidad de iteraciones"
    st.success(
        f"Simulación completada {criterio_txt} – {res['total_iteraciones']} iteraciones "
        f"| reloj final {res.get('hora_final')} "
        f"| tiempo de ejecución {res.get('tiempo_ejecucion_seg', 0):.2f} s "
        f"({len(res['vector'])} filas visibles guardadas)"
    )

# Si todavía no se simuló, se muestra un mensaje inicial y se corta la ejecución.
if "res" not in st.session_state:
    st.info("Configurá los parámetros en el panel izquierdo y presioná **Simular**.")
    st.stop()

# Recupera los resultados guardados.
res: dict = st.session_state["res"]
j = st.session_state["j"]
i = st.session_state["i"]

# Crea las pestañas de visualización.
tab1, tab2, tab3, tab4 = st.tabs([
    "Vector de Estado",
    "Métricas",
    "Controles Periódicos",
    "Runge-Kutta"
])

# ── TAB 1: Vector de Estado ───────────────────────────────────────────────────
with tab1:
    # Vector visible generado por la simulación.
    # Ya viene filtrado desde simular(), por lo tanto no se hace iloc sobre un DataFrame gigante.
    df = res["vector"]
    total = res.get("total_iteraciones", len(df))

    # Informa el rango solicitado y cuántas filas se conservaron realmente.
    st.markdown(
        f"**Total de iteraciones simuladas:** {total}  |  "
        f"**Reloj final:** {res.get('hora_final', '-')}  |  "
        f"**Filas visibles guardadas:** {len(df)}  |  "
        f"Rango solicitado: **{j}** a **{j+i-1}**"
    )

    # Como df ya está filtrado, el subset visible es directamente el vector recibido.
    subset = df

    # Elimina columnas de clientes que no tienen ningún dato en este subset.
    cols_cli = [c for c in subset.columns if re.match(r"C\d+ (ID|ESTADO|Tiempo Llegada|TIPO Pedido|SALON|Tiempo Salida)", c)]
    cols_cli_vacias = [c for c in cols_cli if subset[c].isna().all()]
    subset = subset.drop(columns=cols_cli_vacias)

    # Última fila, correspondiente al fin de simulación.
    ultima = res.get("ultima_fila", pd.DataFrame())
    if ultima.empty and not df.empty:
        ultima = df.iloc[[-1]]

    # Genera los grupos de columnas, incluyendo los clientes vivos que aparezcan
    # dentro del intervalo seleccionado.
    grupos_subset = grupos_con_clientes(subset)

    st.subheader("Iteraciones seleccionadas")
    st.markdown(render_tabla_multinivel(subset.fillna(""), grupos_subset), unsafe_allow_html=True)

    st.subheader("Última fila (fin de simulación)")
    # En la última fila se ocultan columnas temporales de RNDs, porque el enunciado
    # permite no mostrar objetos/variables temporales en la fila final.
    cols_temp = [
        "RND 1 Llegada", "RND 2 Llegada", "Tiempo Llegada 1", "Tiempo Llegada 2",
        "RND Atencion Caja", "Duracion Atencion En Caja",
        "RND Tipo Pedido", "RND A", "A", "TIEMPO PREPARACION PEDIDO LOCAL",
        "RND 1 Llevar", "RND 2 Llevar", "VA 1 Llevar", "VA 2 Llevar",
        "RND Salon",
        "RND 1 Comida Rojo", "RND 2 Comida Rojo", "tiempo de permanencia rojo 1", "tiempo de permanencia rojo 2",
        "RND 1 Comida Azul", "RND 2 Comida Azul", "tiempo de permanencia azul 1", "tiempo de permanencia azul 2",
    ]

    # Para la última fila se eliminan las columnas temporales.
    ultima_visible = ultima.drop(columns=cols_temp, errors="ignore")

    # Se regeneran los grupos usando solamente las columnas que quedaron visibles.
    grupos_ultima = grupos_con_clientes(ultima_visible)

    st.markdown(render_tabla_multinivel(ultima_visible.fillna(""), grupos_ultima), unsafe_allow_html=True)

# ── TAB 2: Métricas ───────────────────────────────────────────────────────────
with tab2:
    # Muestra las métricas finales en tarjetas.
    # Estos contadores se siguen calculando en res["metricas"],
    # pero no se muestran en esta pestaña para dejar el resumen más limpio.
    m = res["metricas"]
    metricas_ocultas = {
        "Prom. tiempo cola mostrador (seg)",
        "Clientes finalizados",
        "Para llevar",
        "En local",
        "Salón rojo",
        "Salón azul",
    }
    items = [(nombre, val) for nombre, val in m.items() if nombre not in metricas_ocultas]

    def mostrar_metrica(nombre, val):
        """Renderiza una métrica manteniendo el formato numérico usado en la app."""
        if isinstance(val, float):
            st.metric(nombre, f"{val:.2f}")
        else:
            st.metric(nombre, val)

    # Distribución visual solicitada:
    # fila 1: métricas 1 y 2;
    # fila 2: métricas 3 y 4;
    # fila 3: métricas 5 y 6;
    # fila 4: métricas 7 y 8.
    filas_metricas = [
        items[0:2],
        items[2:4],
        items[4:6],
        items[6:8],
    ]

    for fila in filas_metricas:
        if not fila:
            continue
        cols = st.columns(len(fila))
        for col, (nombre, val) in zip(cols, fila):
            with col:
                mostrar_metrica(nombre, val)

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