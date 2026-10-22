"""
Calorimetria: misura del calore latente di fusione del ghiaccio.

Inserisci i tuoi dati nella sezione "DATI" (i valori presenti sono SOLO di esempio).
Unità: masse in g, temperature in °C, calore latente in cal/g.

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
TG = 0.0          # °C, temperatura del ghiaccio (fusione)

LAMBDA_ATTESO = 79.7  # cal/g, valore di riferimento

# ----------------------------------------------------------------------------
# DATI  (sostituisci con i tuoi)
# ----------------------------------------------------------------------------
# mg (ghiaccio), ma (acqua calda), Ta, Tf
misure_lambda = [
    dict(ma=200.3, Ta=55.0, mg=70.4, Tg=0.4, Tf=22.4),
    dict(ma=198.3, Ta=54.8, mg=80.8, Tg=0.6, Tf=21.1)
]


# ----------------------------------------------------------------------------
# FORMULA
# ----------------------------------------------------------------------------
def calore_latente(mg, ma, Ta, Tf, Me=ME, Tg=TG):
    """Eq. (4): lambda = ca [(ma + Me)(Ta - Tf) - mg (Tf - Tg)] / mg"""
    return CA * ((ma + Me) * (Ta - Tf) - mg * (Tf - Tg)) / mg


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
    alto = np.mean([calore_latente(**m, Me=Me + sigma_Me) for m in misure])
    basso = np.mean([calore_latente(**m, Me=Me - sigma_Me) for m in misure])
    return abs(alto - basso) / 2


# ----------------------------------------------------------------------------
# ANALISI
# ----------------------------------------------------------------------------
def analizza(misure, Me=ME, sigma_Me=SIGMA_ME, atteso=None):
    print("\n=== calore latente di fusione del ghiaccio ===")
    valori, sigme_stat = [], []
    for i, m in enumerate(misure, 1):
        v, s, contr = propaga(lambda **kw: calore_latente(**kw, Me=Me), m, sigme_lettura(m))
        _, _, c_me = propaga(lambda Me_: calore_latente(**m, Me=Me_),
                             {"Me_": Me}, {"Me_": sigma_Me})
        dominante = max(contr, key=contr.get)
        print(f"  misura {i}: {v:.2f} ± {s:.2f} (stat) ± {c_me['Me_']:.2f} (Me) cal/g"
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

    print(f"  media pesata: {media:.2f} ± {s_media:.2f} (stat) ± {s_sys:.2f} (Me)"
          f" = {media:.2f} ± {s_tot:.2f} cal/g  ({media*4.186:.0f} ± {s_tot*4.186:.0f} J/g)")

    if n >= 2:
        chi2 = np.sum(((valori - media) / sigme_stat) ** 2)
        print(f"  chi2 = {chi2:.2f} con {n-1} gdl  "
              f"({'dispersione compatibile' if chi2 < 2*(n-1) else 'dispersione > errori stimati'})")

    if atteso is not None:
        z = abs(media - atteso) / s_tot
        print(f"  atteso {atteso} cal/g -> scarto = {z:.1f} sigma")
    return media, s_tot


if __name__ == "__main__":
    print(f"Massa equivalente assunta: Me = {ME:.1f} ± {SIGMA_ME:.1f} g")
    analizza(misure_lambda, atteso=LAMBDA_ATTESO)