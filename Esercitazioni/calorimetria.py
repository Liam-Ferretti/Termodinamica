"""
Calorimetria: misura del calore specifico di alcuni materiali.

Inserisci i tuoi dati nella sezione "DATI" (i valori presenti sono SOLO di esempio).
Unità: masse in g, temperature in °C, calori specifici in cal/(g K).

Nota sulle incertezze: l'errore su M_e è COMUNE a tutte le misure (sistematico),
quindi non si riduce mediando. Lo script lo tratta separatamente dall'errore
"statistico" (bilancia + termometro):
    - sigma_stat : propagazione degli errori di lettura, si riduce con la media
    - sigma_sys  : effetto di M_e +- sigma_Me, uguale per tutte le misure
"""

import numpy as np

# ----------------------------------------------------------------------------
# PARAMETRI SPERIMENTALI
# ----------------------------------------------------------------------------
CA = 1.0          # cal/(g K), calore specifico dell'acqua
ME = 25.0         # g, massa equivalente del thermos
SIGMA_ME = 5.0    # g
SIGMA_M = 0.1     # g, risoluzione bilancia (modifica se diversa)
SIGMA_T = 0.1     # °C, incertezza termometro (es. mezza divisione)

# Valori di riferimento (letteratura, indicativi)
C_ATTESI = {"alluminio": 0.215, "ottone": 0.092, "granito": 0.19, "okite": 0.20}

# ----------------------------------------------------------------------------
# DATI  (sostituisci con i tuoi)
# ----------------------------------------------------------------------------
# m1 (corpo), T1 (corpo caldo), ma (acqua fredda), Ta, Tf
# masse dei campioni: NON presenti nelle tabelle, inseriscile tu
M_OKITE = 137.9      # g
M_ALLUMINIO = 78.9  # g

# ----------------------------------------------------------------------------
# DATI
# ----------------------------------------------------------------------------
# m1 (corpo), ma (acqua fredda), Ta, T1 (corpo caldo), Tf
M_OKITE = 137.9      # g
M_ALLUMINIO = 78.9   # g
M_OTTONE = 194.8   # g
M_GRANITO = 38.0   # g

misure_c = {
    "okite": [
        dict(m1=M_OKITE, ma=200.5, Ta=26.8, T1=64.6, Tf=30.6),
        dict(m1=M_OKITE, ma=199.7, Ta=25.0, T1=62.8, Tf=29.0),
    ],
    "alluminio": [
        dict(m1=M_ALLUMINIO, ma=200.5, Ta=24.9, T1=61.8, Tf=27.4),
        dict(m1=M_ALLUMINIO, ma=202.5, Ta=25.5, T1=64.4, Tf=28.3),
        dict(m1=M_ALLUMINIO, ma=204.7, Ta=24.4, T1=63.2, Tf=27.1),
        dict(m1=M_ALLUMINIO, ma=200.8, Ta=23.8, T1=63.9, Tf=26.7),
        dict(m1=M_ALLUMINIO, ma=201.1, Ta=25.0, T1=63.5, Tf=27.9),
    ],
    "ottone": [
        dict(m1=M_OTTONE, ma=202.8, Ta=24.8, T1=62.8, Tf=27.6),
        dict(m1=M_OTTONE, ma=199.6, Ta=25.0, T1=61.4, Tf=28.0),
    ],
    "granito": [
        dict(m1=M_GRANITO, ma=201.5, Ta=25.2, T1=64.2, Tf=26.5),
        dict(m1=M_GRANITO, ma=200.7, Ta=24.8, T1=66.0, Tf=26.1),
        dict(m1=M_GRANITO, ma=200.8, Ta=25.6, T1=65.0, Tf=27.6),
        dict(m1=M_GRANITO, ma=201.0, Ta=25.0, T1=65.8, Tf=26.9)
    ],
}


# ----------------------------------------------------------------------------
# FORMULA
# ----------------------------------------------------------------------------
def calore_specifico(m1, T1, ma, Ta, Tf, Me=ME):
    """Eq. (3): c1 = ca (ma + Me)(Tf - Ta) / [m1 (T1 - Tf)]"""
    return CA * (ma + Me) * (Tf - Ta) / (m1 * (T1 - Tf))


# ----------------------------------------------------------------------------
# PROPAGAZIONE DEGLI ERRORI
# ----------------------------------------------------------------------------
def propaga(f, valori, sigme, h=1e-6):
    """
    Propagazione per variabili indipendenti con derivate numeriche centrali.
    Restituisce (valore, sigma, contributi) con contributi[k] = |df/dx_k| * sigma_k.
    """
    f0 = f(**valori)
    contributi = {}
    for k, s in sigme.items():
        if s == 0:
            continue
        step = h * max(abs(valori[k]), 1.0)
        su = dict(valori, **{k: valori[k] + step})
        giu = dict(valori, **{k: valori[k] - step})
        derivata = (f(**su) - f(**giu)) / (2 * step)
        contributi[k] = abs(derivata) * s
    sigma = np.sqrt(sum(c**2 for c in contributi.values()))
    return f0, sigma, contributi


def sigme_lettura(misura):
    """Errori di lettura: SIGMA_M sulle masse, SIGMA_T sulle temperature."""
    return {k: (SIGMA_M if k.startswith("m") else SIGMA_T) for k in misura}


def sistematico_Me(misure, Me, sigma_Me):
    """Effetto di M_e sulla media: semi-differenza fra le medie a Me +- sigma_Me."""
    alto = np.mean([calore_specifico(**m, Me=Me + sigma_Me) for m in misure])
    basso = np.mean([calore_specifico(**m, Me=Me - sigma_Me) for m in misure])
    return abs(alto - basso) / 2


# ----------------------------------------------------------------------------
# ANALISI
# ----------------------------------------------------------------------------
def analizza(materiale, misure, Me=ME, sigma_Me=SIGMA_ME, atteso=None):
    print(f"\n=== calore specifico {materiale} ===")
    valori, sigme_stat = [], []
    for i, m in enumerate(misure, 1):
        v, s, contr = propaga(lambda **kw: calore_specifico(**kw, Me=Me), m, sigme_lettura(m))
        _, _, c_me = propaga(lambda Me_: calore_specifico(**m, Me=Me_),
                             {"Me_": Me}, {"Me_": sigma_Me})
        dominante = max(contr, key=contr.get)
        print(f"  misura {i}: {v:.4f} ± {s:.4f} (stat) ± {c_me['Me_']:.4f} (Me) cal/(g K)"
              f"   [errore di lettura dominante: {dominante}]")
        valori.append(v)
        sigme_stat.append(s)

    valori, sigme_stat = np.array(valori), np.array(sigme_stat)
    n = len(valori)

    # media pesata con gli errori statistici
    w = 1 / sigme_stat**2
    media = np.sum(w * valori) / np.sum(w)
    s_media = 1 / np.sqrt(np.sum(w))
    s_sys = sistematico_Me(misure, Me, sigma_Me)
    s_tot = np.hypot(s_media, s_sys)

    print(f"  media pesata: {media:.4f} ± {s_media:.4f} (stat) ± {s_sys:.4f} (Me)"
          f" = {media:.4f} ± {s_tot:.4f} cal/(g K)")

    if n >= 3:
        dev_std = np.std(valori, ddof=1)
        print(f"  dispersione: dev. std = {dev_std:.4f}, dev. std media = {dev_std/np.sqrt(n):.4f}"
              f"  |  errore propagato medio = {np.mean(sigme_stat):.4f}")
        chi2 = np.sum(((valori - media) / sigme_stat) ** 2)
        print(f"  chi2 = {chi2:.2f} con {n-1} gdl  "
              f"({'dispersione compatibile' if chi2 < 2*(n-1) else 'dispersione > errori stimati'})")

    if atteso is not None:
        z = abs(media - atteso) / s_tot
        print(f"  atteso {atteso} cal/(g K) -> scarto = {z:.1f} sigma")
    return media, s_tot


if __name__ == "__main__":
    print(f"Massa equivalente assunta: Me = {ME:.1f} ± {SIGMA_ME:.1f} g")
    for materiale, misure in misure_c.items():
        analizza(materiale, misure, atteso=C_ATTESI.get(materiale))