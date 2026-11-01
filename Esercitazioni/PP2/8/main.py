import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def comma_to_float(x):
    if isinstance(x, bytes):
        x = x.decode('latin1')
    return float(x.replace(',', '.'))

LSB        = 0.005  # risoluzione del sensore [kPa]
window_s   = 3.0        # finestra mobile [s]
soglia     = 1.1 * LSB  # soglia sulla std [kPa]
min_dur_s  = 5.0        # durata minima di uno stallo [s]
max_gap_s  = 3.0        # buco massimo tra due zone da unire [s]
tol_merge  = 7 * LSB   # differenza massima di livello per unire [kPa]

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



V_diff = 10
V_0 = 54
p_0 = 100.97

pressioni = np.zeros(6)

m = []
c = []

for i in range(1, 5):
        
    file = f"8/8-{i}"
    t, P = np.loadtxt(f"{file}.txt", unpack=True, skiprows=2, encoding='latin1',
                      converters={0: comma_to_float, 1: comma_to_float})

    zone = trova_stalli(t, P)

    plt.figure(figsize=(10, 6))
    plt.plot(t, P, color='black', alpha=0.6, label='Dati sperimentali')

    print(f"\nTest {i}")
    for j, (s, e) in enumerate(zone, 1):
        seg = P[s:e]
        plt.plot(t[s:e], seg, linewidth=3, label=f'Stallo {j}')
        print(f"  Stallo {j}: {t[s]:7.2f}–{t[e-1]:7.2f} s  "
              f"P = {seg.mean():.4f} ± {seg.std(ddof=1):.4f} kPa")
        
        pressioni[j - 1] = (seg.mean())

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Analisi pressione - Test {i}")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Pressione [kPa]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{file}_analisi_transitorio.png", dpi=300)
    plt.show()
    
    # CALCOLO DEI DELTA p E DELTA V
    delta_p = np.zeros_like(pressioni)
    delta_V = np.zeros_like(pressioni)

    for j in range(len(pressioni)):
        delta_p[j] = -(p_0 - pressioni[j])
        delta_V[j] = -(V_0 - j * V_diff)
        print("\nDifferenze di pressione e volume")
        print(f"{delta_p[j]:.4f} | {delta_V[j]:.2f}")
        
    x = delta_p / p_0
    y = delta_V / V_0
        
    a, b = np.polyfit(x, y, 1)
    
    m.append(a)
    c.append(b)

    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = a * x_fit + b
    
    plt.scatter(x, y, label="Dati sperimentali")
    plt.plot(x_fit, y_fit, color='red', label="Fit lineare")

    plt.title("Analisi confronto pressione relative vs volume relativo")
    plt.xlabel("Differenza di pressione relativa")
    plt.ylabel("Differenza di volume relativo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{file}_fit_lineare.png", dpi=300)

    plt.show()
    
plt.scatter(m, c, label="Dati sperimentali")

plt.title("Confronto dei coefficienti lineari")
plt.xlabel("m")
plt.ylabel("c")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()