import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import info

def comma_to_float(x):
    if isinstance(x, bytes):
        x = x.decode('latin1')
    return float(x.replace(',', '.'))

LSB        = info.risoluzione_pressione_CI6534  # risoluzione del sensore [kPa]
window_s   = 3.0        # finestra mobile [s]
soglia     = 1.1 * LSB  # soglia sulla std [kPa]
min_dur_s  = 5.0        # durata minima di uno stallo [s]
max_gap_s  = 3.0        # buco massimo tra due zone da unire [s]
tol_merge  = 7 * LSB    # differenza massima di livello per unire [kPa]

def trova_stalli(t, P):
    dt = np.median(np.diff(t))
    w = int(round(window_s / dt))
    half = w // 2

    stds = np.std(sliding_window_view(P, w), axis=1, ddof=1)

    is_st = np.zeros(len(P), dtype=bool)
    is_st[half:half + len(stds)] = stds < soglia

    d = np.diff(np.r_[0, is_st.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)

    zone = []
    for s, e in zip(starts, ends):
        if zone:
            ps, pe = zone[-1]
            vicine = (t[s] - t[pe - 1]) <= max_gap_s
            simili = abs(np.median(P[s:e]) - np.median(P[ps:pe])) < tol_merge
            if vicine and simili:
                zone[-1][1] = e
                continue
        zone.append([s, e])

    return [(s, e) for s, e in zone if t[e - 1] - t[s] >= min_dur_s]


V_diff = 10      # ml, compressione tra uno stallo e il successivo
V_0 = 54         # ml
p_0 = 100.97     # kPa, pressione atmosferica (barometro dell'aula)

m, sigma_m = [], []
q, sigma_q = [], []

for i in range(1, 5):

    file = f"8/8-{i}"
    t, P = np.loadtxt(f"{file}.txt", unpack=True, skiprows=2, encoding='latin1',
                      converters={0: comma_to_float, 1: comma_to_float})

    zone = trova_stalli(t, P)
    n_stalli = len(zone)

    pressioni = np.zeros(n_stalli)
    sigma_a_p = np.zeros(n_stalli)

    plt.figure(figsize=(10, 6))
    plt.plot(t, P, color='black', alpha=0.6, label='Dati sperimentali')

    print(f"\nTest {i}")
    for j, (s, e) in enumerate(zone):
        seg = P[s:e]
        pressioni[j] = seg.mean()
        # INCERTEZZA DI TIPO A: deviazione standard della media sullo stallo
        sigma_a_p[j] = seg.std(ddof=1) / np.sqrt(len(seg))

        plt.plot(t[s:e], seg, linewidth=3, label=f'Stallo {j + 1}')
        print(f"  Stallo {j + 1}: {t[s]:7.2f}–{t[e-1]:7.2f} s  "
              f"P = {pressioni[j]:.4f} kPa | sigma_A = {sigma_a_p[j]:.5f} kPa")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Analisi pressione - Test {i}")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Pressione [kPa]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{file}_analisi_transitorio.png", dpi=300)
    plt.show()

    # INCERTEZZA TOTALE (A + B). Per B si usa solo la risoluzione:
    # l'offset dell'accuracy viene assorbito dall'intercetta del fit.
    sigma_p = np.sqrt(sigma_a_p**2 + info.risoluzione_pressione_CI6534**2)

    # CALCOLO DEI DELTA p E DELTA V
    # Il CI-6534 misura gia' la pressione relativa all'ambiente -> P = delta p
    delta_p = pressioni
    delta_V = -np.arange(n_stalli) * V_diff   # compressioni: V - V0 < 0

    print("\n  delta p [kPa] | sigma [kPa] | delta V [ml]")
    for dp, sp, dv in zip(delta_p, sigma_p, delta_V):
        print(f"  {dp:9.4f}     | {sp:.5f}     | {dv:6.1f}")

    # Eq. 4: delta p / p0 = -(delta V / V0)
    x = delta_V / V_0
    y = delta_p / p_0
    sigma_y = sigma_p / p_0

    coefficienti, matrice_cov = np.polyfit(x, y, 1, w=1/sigma_y, cov='unscaled')
    a, b = coefficienti
    err_a, err_b = np.sqrt(np.diag(matrice_cov))

    chi2 = np.sum(((y - (a * x + b)) / sigma_y)**2)
    gdl = len(x) - 2

    print(f"\n  m = {a:.4f} ± {err_a:.4f}  (atteso -1)")
    print(f"  q = {b:.5f} ± {err_b:.5f}")
    print(f"  chi2 / gdl = {chi2:.2f} / {gdl}")

    m.append(a); sigma_m.append(err_a)
    q.append(b); sigma_q.append(err_b)

    x_fit = np.linspace(x.min(), x.max(), 100)

    plt.figure(figsize=(10, 6))
    plt.errorbar(x, y, yerr=sigma_y, fmt='o', color='black', markersize=4,
                 ecolor='gray', elinewidth=1, capsize=2, label="Dati sperimentali")
    plt.plot(x_fit, a * x_fit + b, color='red',
             label=f"Fit lineare: m = {a:.3f} ± {err_a:.3f}")
    plt.title(f"Legge di Boyle - Test {i}")
    plt.xlabel(r"$\Delta V / V_0$")
    plt.ylabel(r"$\Delta p / p_0$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{file}_fit_lineare.png", dpi=300)
    plt.show()

# CONFRONTO TRA LE RIPETIZIONI
m, sigma_m = np.array(m), np.array(sigma_m)
q, sigma_q = np.array(q), np.array(sigma_q)

pesi = 1 / sigma_m**2
m_medio = np.sum(pesi * m) / np.sum(pesi)
sigma_m_medio = 1 / np.sqrt(np.sum(pesi))
print(f"\nm medio pesato = {m_medio:.4f} ± {sigma_m_medio:.4f}")
print(f"dispersione tra le prove (std) = {m.std(ddof=1):.4f}")

plt.figure(figsize=(8, 6))
plt.errorbar(m, q, xerr=sigma_m, yerr=sigma_q, fmt='o', capsize=3,
             label="Ripetizioni")
plt.axvline(-1, color='gray', linestyle='--', label="Atteso (m = -1)")
plt.title("Confronto dei coefficienti del fit")
plt.xlabel("m")
plt.ylabel("q")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()