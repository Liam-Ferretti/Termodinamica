import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

import matplotlib.pyplot as plt
import numpy as np
import info

def comma_to_float(x):
    if isinstance(x, bytes):
        x = x.decode('latin1')
    return float(x.replace(',', '.'))


def carica(percorso):
    return np.loadtxt(percorso, unpack=True, skiprows=2, encoding='latin1',
                      converters={0: comma_to_float, 1: comma_to_float})


def trova_inizio(t, pos, durata_stabile=0.5, tol=1e-9):
    """
    Restituisce (idx_massa, idx_inizio):
      idx_massa  -> primo punto in cui la posizione cambia (massa appoggiata)
      idx_inizio -> primo punto dopo il rimbalzo da cui la posizione
                    resta costante per almeno `durata_stabile` secondi
    """
    variati = np.flatnonzero(np.abs(pos - pos[0]) > tol)
    if len(variati) == 0:
        return 0, 0
    idx_massa = variati[0]

    dt = np.median(np.diff(t))
    n = max(int(round(durata_stabile / dt)), 2)

    for k in range(idx_massa, len(pos) - n):
        if np.all(np.abs(pos[k:k + n] - pos[k]) < tol):
            return idx_massa, k

    return idx_massa, idx_massa


def banda_fit(x, cov):
    """Incertezza a 1 sigma della retta y = a x + b in ogni punto x."""
    return np.sqrt(x**2 * cov[0, 0] + cov[1, 1] + 2 * x * cov[0, 1])


# ======================= PARAMETRI =======================
CARTELLA = "9"
g = 9.81  # m/s^2

# --- Bilancia Kern FCB 3K0.1 ---
RIS_MASSA_g = info.risoluzione_massa_FCB3K01    # g
ACC_MASSA_g = info.accuracy_massa_FCB3K01       # g

# --- Pesate ripetute dei singoli pesetti [g] ---
PESATE_g = [
    [83.3, 83.7, 83.8, 83.8],   # pesetto 1
    [84.3, 84.1, 84.2, 84.0],   # pesetto 2
    [83.2, 83.1, 83.2, 83.2],   # pesetto 3
    [82.0, 81.9, 82.0, 82.2],   # pesetto 4
]

medie_pesetti = np.array([np.mean(x) for x in PESATE_g])
sigma_a_pesetti = np.array([np.std(x, ddof=1) / np.sqrt(len(x)) for x in PESATE_g])

# Masse cumulative: nel file k ci sono i pesetti 1..k (il primo file e' senza massa)
masse_cum = [0.0]
sigma_masse_cum = [0.0]
for k in range(1, len(PESATE_g) + 1):
    masse_cum.append(medie_pesetti[:k].sum())
    sigma_stat = np.sqrt(np.sum(sigma_a_pesetti[:k]**2) + k * RIS_MASSA_g**2)
    sigma_acc = k * ACC_MASSA_g      # stessa bilancia: sistematico, si somma linearmente
    sigma_masse_cum.append(np.sqrt(sigma_stat**2 + sigma_acc**2))

print("Pesetti:")
for j, (mj, sj) in enumerate(zip(medie_pesetti, sigma_a_pesetti), 1):
    print(f"  pesetto {j}: {mj:.3f} g (sigma_A = {sj:.3f} g)")

print("Masse cumulative:")
for k, (mk, sk) in enumerate(zip(masse_cum, sigma_masse_cum)):
    print(f"  file {k + 1}: m = {mk:.3f} ± {sk:.3f} g")

# --- Pistone ---
D_PISTONE = 32.5e-3               # m   <-- VERIFICARE
SIGMA_D   = 0.1e-3                # m   <-- incertezza sul diametro
S         = np.pi * (D_PISTONE / 2)**2
SIGMA_S   = S * 2 * SIGMA_D / D_PISTONE

# --- Sensore di posizione CI-6538 ---
# risoluzione nominale da info (usata solo se dai dati non si ricava il passo)
RIS_POS_NOMINALE = info.risoluzione_posizione_CI6538 * 1e-3      # m
# info.accuracy_posizione_CI6538 = 0 e' un offset assoluto: non influisce sulla pendenza.
# Per alpha conta solo un errore di scala (es. sul raggio della puleggia):
ACC_REL_POSIZIONE = 0.0           # <-- es. sigma_raggio / raggio, se nota

# --- Sensore di pressione CI-6534 (solo per il controllo) ---
SIGMA_B_P = info.sigma_b_pressione_CI6534     # kPa (risoluzione + accuracy)

M3S_TO_CM3MIN = 1e6 * 60          # da m^3/s a cm^3/min

SERIE = {
    "Solo cilindro-pistone":   [f"9-{i}" for i in range(1, 6)],
    "Con cilindro di metallo": [f"9-2-{i}" for i in range(1, 6)],
}

# stessi pesetti per entrambe le serie
MASSE_g = {nome: masse_cum for nome in SERIE}
SIGMA_MASSE_g = {nome: sigma_masse_cum for nome in SERIE}


# ======================= ANALISI DI UN FILE =======================
def analizza_file(nome, con_massa, massa_g, sigma_massa_g):
    t, h = carica(f"{CARTELLA}/POS-{nome}.txt")
    t_p, p = carica(f"{CARTELLA}/PRES-{nome}.txt")

    if con_massa:
        idx_massa, idx_inizio = trova_inizio(t, h)
    else:
        idx_massa, idx_inizio = None, 0

    t_sel, h_sel = t[idx_inizio:], h[idx_inizio:]

    # --- POSIZIONE: risoluzione dal passo dei dati, incertezza di quantizzazione
    passi = np.diff(np.unique(np.round(h, 12)))
    lsb = passi.min() if len(passi) > 0 else RIS_POS_NOMINALE
    sigma_h = lsb / np.sqrt(12)

    # FIT h(t) = alpha * t + q
    (alpha, q), cov_h = np.polyfit(t_sel, h_sel, 1,
                                   w=np.full(len(t_sel), 1 / sigma_h),
                                   cov='unscaled')
    sigma_alpha_ris = np.sqrt(cov_h[0, 0])
    # l'accuracy di scala del sensore agisce come errore relativo sulla pendenza
    sigma_alpha_acc = abs(alpha) * ACC_REL_POSIZIONE
    sigma_alpha = np.sqrt(sigma_alpha_ris**2 + sigma_alpha_acc**2)

    # --- MASSA e dp = m g / S (incertezza puntuale, senza S: S e' trattata a parte)
    m = massa_g * 1e-3
    sigma_m = sigma_massa_g * 1e-3
    dp = m * g / S / 1e3                     # kPa
    sigma_dp = g * sigma_m / S / 1e3         # kPa

    # --- BETA = alpha * S (incertezza puntuale, senza S)
    beta = alpha * S * M3S_TO_CM3MIN
    sigma_beta = sigma_alpha * S * M3S_TO_CM3MIN

    # --- CONTROLLO: pressione misurata dal sensore (gia' relativa all'ambiente)
    maschera = t_p >= t[idx_inizio]
    t_p_sel, p_sel = t_p[maschera], p[maschera]
    p_mis = p_sel.mean()
    sigma_p_mis = np.sqrt((p_sel.std(ddof=1) / np.sqrt(len(p_sel)))**2 + SIGMA_B_P**2)

    print(f"  {nome}: m = {massa_g:.2f} ± {sigma_massa_g:.2f} g | "
          f"mg/S = {dp:.4f} ± {sigma_dp:.4f} kPa | "
          f"p misurata = {p_mis:.4f} ± {sigma_p_mis:.4f} kPa")
    print(f"         alpha = ({alpha*1e6:.3f} ± {sigma_alpha*1e6:.3f}) um/s "
          f"[ris {sigma_alpha_ris*1e6:.3f}, acc {sigma_alpha_acc*1e6:.3f}] | "
          f"beta = {beta:.4f} ± {sigma_beta:.4f} cm^3/min")

    # ======================= GRAFICI =======================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Posizione
    if con_massa:
        ax1.plot(t, h * 1e3, color='gray', alpha=0.5, label='Dati completi')
        ax1.axvline(t[idx_massa], color='red', linestyle='--',
                    label=f'Massa appoggiata (t = {t[idx_massa]:.2f} s)')
        ax1.axvline(t[idx_inizio], color='blue', linestyle='--',
                    label=f'Inizio analisi (t = {t[idx_inizio]:.2f} s)')

    ax1.fill_between(t_sel, (h_sel - sigma_h) * 1e3, (h_sel + sigma_h) * 1e3,
                     color='black', alpha=0.15, step='mid',
                     label=f'Incertezza dati ±{sigma_h*1e3:.3f} mm (LSB/√12)')
    ax1.step(t_sel, h_sel * 1e3, where='mid', color='black', linewidth=1,
             label="Dati per l'analisi")

    h_fit = alpha * t_sel + q
    sigma_fit = banda_fit(t_sel, cov_h)
    ax1.plot(t_sel, h_fit * 1e3, color='orange',
             label=f'Fit: alpha = ({alpha*1e6:.2f} ± {sigma_alpha*1e6:.2f}) um/s')
    ax1.fill_between(t_sel, (h_fit - sigma_fit) * 1e3, (h_fit + sigma_fit) * 1e3,
                     color='orange', alpha=0.35, label='Banda del fit (1σ)')

    ax1.set_ylabel("Posizione h [mm]")
    ax1.set_title(f"Tenuta cilindro-pistone [{nome}] - m = {massa_g:.2f} ± {sigma_massa_g:.2f} g")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Pressione
    ax2.plot(t_p, p, color='gray', alpha=0.5)
    ax2.fill_between(t_p_sel, p_sel - SIGMA_B_P, p_sel + SIGMA_B_P,
                     color='black', alpha=0.15,
                     label=f'Incertezza lettura ±{SIGMA_B_P:.3f} kPa (ris. + acc.)')
    ax2.plot(t_p_sel, p_sel, color='black', linewidth=1)
    ax2.axhline(p_mis, color='green', linestyle='--',
                label=f'p media = {p_mis:.3f} ± {sigma_p_mis:.3f} kPa')
    ax2.axhspan(p_mis - sigma_p_mis, p_mis + sigma_p_mis, color='green', alpha=0.2)
    ax2.set_xlabel("Tempo [s]")
    ax2.set_ylabel("Pressione relativa [kPa]")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{CARTELLA}/{nome}_analisi.png", dpi=300)
    plt.show()

    return dp, sigma_dp, beta, sigma_beta, p_mis, sigma_p_mis


# ======================= FIT CON ERRORI SU x E y =======================
def fit_lineare_xy(x, sx, y, sy, n_iter=10):
    """
    Fit y = a x + b con il metodo della varianza efficace:
    l'incertezza su x viene propagata sulle ordinate, sy_eff^2 = sy^2 + (a sx)^2.
    """
    a, b = np.polyfit(x, y, 1, w=1 / sy)
    for _ in range(n_iter):
        s_eff = np.sqrt(sy**2 + (a * sx)**2)
        (a, b), cov = np.polyfit(x, y, 1, w=1 / s_eff, cov='unscaled')
    s_eff = np.sqrt(sy**2 + (a * sx)**2)
    chi2 = np.sum(((y - (a * x + b)) / s_eff)**2)
    return a, b, cov, chi2, s_eff


# ======================= ANALISI DELLE DUE SERIE =======================
risultati = {}

for nome_serie, files in SERIE.items():
    masse = MASSE_g[nome_serie]
    sigma_masse = SIGMA_MASSE_g[nome_serie]

    print(f"\n{nome_serie}")
    dati = np.array([analizza_file(nome, con_massa=(k != 0),
                                   massa_g=masse[k], sigma_massa_g=sigma_masse[k])
                     for k, nome in enumerate(files)])
    dp, sigma_dp, beta, sigma_beta, p_mis, sigma_p_mis = dati.T

    # FIT beta = omega * dp + c
    omega, c, cov_fit, chi2, sigma_eff = fit_lineare_xy(dp, sigma_dp, beta, sigma_beta)
    sigma_omega_stat, sigma_c = np.sqrt(np.diag(cov_fit))

    # S compare in beta (∝ S) e in dp (∝ 1/S) -> omega ∝ S^2: errore sistematico comune
    sigma_omega_S = abs(omega) * 2 * SIGMA_S / S
    sigma_omega = np.sqrt(sigma_omega_stat**2 + sigma_omega_S**2)

    print(f"  Contributi a sigma_beta_eff: sigma_beta = {np.round(sigma_beta, 5)}, "
          f"omega*sigma_dp = {np.round(abs(omega) * sigma_dp, 5)} cm^3/min")
    print(f"  omega = {omega:.4f} ± {sigma_omega:.4f} cm^3/(kPa min) "
          f"[fit {sigma_omega_stat:.4f}, superficie S {sigma_omega_S:.4f}]")
    print(f"  c     = {c:.4f} ± {sigma_c:.4f} cm^3/min")
    print(f"  chi2 / gdl = {chi2:.2f} / {len(dp) - 2}")

    # controllo: p misurata - mg/S dovrebbe essere ~ costante (peso del pistone / S)
    diff = p_mis - dp
    print(f"  p misurata - mg/S = {np.round(diff, 3)} kPa "
          f"(media {diff.mean():.3f} kPa -> m_pistone ≈ {diff.mean()*1e3*S/g*1e3:.1f} g)")

    risultati[nome_serie] = dict(dp=dp, beta=beta, sigma_eff=sigma_eff,
                                 omega=omega, c=c, cov_fit=cov_fit,
                                 sigma_omega=sigma_omega,
                                 sigma_omega_stat=sigma_omega_stat)


# ======================= CONFRONTO =======================
plt.figure(figsize=(10, 6))
for (nome_serie, r), col in zip(risultati.items(), ['tab:blue', 'tab:red']):
    plt.errorbar(r['dp'], r['beta'], yerr=r['sigma_eff'],
                 fmt='o', color=col, capsize=3,
                 label=f"{nome_serie} (σ efficace)")

    x_fit = np.linspace(0, r['dp'].max() * 1.05, 200)
    y_fit = r['omega'] * x_fit + r['c']
    s_fit = banda_fit(x_fit, r['cov_fit'])

    plt.plot(x_fit, y_fit, color=col, linestyle='--',
             label=f"omega = {r['omega']:.3f} ± {r['sigma_omega']:.3f} cm³/(kPa·min)")
    plt.fill_between(x_fit, y_fit - s_fit, y_fit + s_fit, color=col, alpha=0.2)

plt.title("Perdita di volume in funzione della sovrapressione")
plt.xlabel(r"$\Delta p = mg/S$ [kPa]")
plt.ylabel(r"$\beta = \alpha \cdot S$ [cm³/min]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{CARTELLA}/confronto_omega.png", dpi=300)
plt.show()

(n1, r1), (n2, r2) = risultati.items()
# stesso pistone per entrambe le serie: l'incertezza su S e' comune e non entra nel confronto
z = abs(r1['omega'] - r2['omega']) / np.sqrt(r1['sigma_omega_stat']**2 + r2['sigma_omega_stat']**2)
print(f"\nCompatibilita' tra gli omega: z = {z:.2f} sigma")