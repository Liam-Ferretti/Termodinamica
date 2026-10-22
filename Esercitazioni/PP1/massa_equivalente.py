"""
Calorimetria: misura della massa equivalente (in acqua) del thermos.

Inserisci i tuoi dati nella sezione "DATI" (i valori presenti sono SOLO di esempio).
Unità: masse in g, temperature in °C.

Bilancio (acqua calda m1 nel thermos, acqua fredda m2 versata dentro):
    (m1 + Me)(T1 - Tf) = m2 (Tf - T2)
    =>  Me = m2 (T2 - Tf) / (Tf - T1) - m1
"""

import numpy as np

# ----------------------------------------------------------------------------
# PARAMETRI SPERIMENTALI
# ----------------------------------------------------------------------------
SIGMA_M = 0.1     # g, risoluzione bilancia (modifica se diversa)
SIGMA_T = 0.1     # °C, incertezza termometro (es. mezza divisione)

ME_ASSUNTA = 25.0       # g, valore usato nelle altre misure
SIGMA_ME_ASSUNTA = 5.0  # g

# ----------------------------------------------------------------------------
# DATI  (sostituisci con i tuoi)
# ----------------------------------------------------------------------------
# m1, T1 (acqua calda, nel thermos), m2, T2 (acqua fredda, versata), Tf
misure_Me = [
    dict(m1=102.1, T1=47.6, m2=100.8, T2=6.0, Tf=28.0),
    dict(m1=99.2, T1=50.8, m2=100.3, T2=4.8, Tf=26.4) 
]


# ----------------------------------------------------------------------------
# FORMULA
# ----------------------------------------------------------------------------
def massa_equivalente(m1, T1, m2, T2, Tf):
    """Me = m2 (T2 - Tf) / (Tf - T1) - m1"""
    return m2 * (T2 - Tf) / (Tf - T1) - m1


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


# ----------------------------------------------------------------------------
# ANALISI
# ----------------------------------------------------------------------------
def analizza(misure, atteso=ME_ASSUNTA, sigma_atteso=SIGMA_ME_ASSUNTA):
    print("\n=== massa equivalente del thermos ===")
    valori, sigme = [], []
    for i, m in enumerate(misure, 1):
        v, s, contr = propaga(massa_equivalente, m, sigme_lettura(m))
        dominante = max(contr, key=contr.get)
        print(f"  misura {i}: Me = {v:.1f} ± {s:.1f} g"
              f"   [errore di lettura dominante: {dominante}]")
        valori.append(v)
        sigme.append(s)

    valori, sigme = np.array(valori), np.array(sigme)
    n = len(valori)

    # media pesata
    w = 1 / sigme**2
    media = np.sum(w * valori) / np.sum(w)
    s_media = 1 / np.sqrt(np.sum(w))
    print(f"  media pesata: Me = {media:.1f} ± {s_media:.1f} g")

    if n >= 2:
        chi2 = np.sum(((valori - media) / sigme) ** 2)
        print(f"  chi2 = {chi2:.2f} con {n-1} gdl  "
              f"({'dispersione compatibile' if chi2 < 2*(n-1) else 'dispersione > errori stimati'})")

    z = abs(media - atteso) / np.hypot(s_media, sigma_atteso)
    print(f"  confronto con Me assunta ({atteso:.0f} ± {sigma_atteso:.0f}) g -> scarto = {z:.1f} sigma")
    return media, s_media


if __name__ == "__main__":
    analizza(misure_Me)