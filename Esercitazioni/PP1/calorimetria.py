import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# PARAMETRI SPERIMENTALI
# ----------------------------------------------------------------------------
CA = 1.0          # cal/(g K), calore specifico dell'acqua
ME = 25.0         # g, massa equivalente del thermos
SIGMA_ME = 5.0    # g
SIGMA_M = np.sqrt((0.1/np.sqrt(12))**2 + (0.3)**2)     # g, risoluzione bilancia + linearity
SIGMA_T = np.sqrt((0.2/np.sqrt(12))**2 + (1)**2)  # °C, incertezza termometro + accuracy del termometro

C_ATTESI = {"okite": 0.20, "alluminio": 0.215, "ottone": 0.092, "granito": 0.19}  # cal/(g K), valori di riferimento

M_OKITE = 137.9      # g
M_ALLUMINIO = 78.9   # g
M_OTTONE = 194.8     # g
M_GRANITO = 38.0     # g

materiale = ["okite",
             "okite",
             "alluminio",
             "alluminio",
             "alluminio",
             "alluminio",
             "alluminio",
             "ottone",
             "ottone",
             "granito",
             "granito",
             "granito",
             "granito"]

m1 = [M_OKITE,
      M_OKITE,
      M_ALLUMINIO,
      M_ALLUMINIO,
      M_ALLUMINIO,
      M_ALLUMINIO,
      M_ALLUMINIO,
      M_OTTONE,
      M_OTTONE,
      M_GRANITO,
      M_GRANITO,
      M_GRANITO,
      M_GRANITO]

ma = [200.5,
      199.7,
      200.5,
      202.5,
      204.7,
      200.8,
      201.1,
      202.8,
      199.6,
      201.5,
      200.7,
      200.8,
      201.0]

Ta = [26.8,
      25.0,
      24.9,
      25.5,
      24.4,
      23.8,
      25.0,
      24.8,
      25.0,
      25.2,
      24.8,
      25.6,
      25.0]

T1 = [64.6,
      62.8,
      61.8,
      64.4,
      63.2,
      63.9,
      63.5,
      62.8,
      61.4,
      64.2,
      66.0,
      65.0,
      65.8]

Tf = [30.6,
      29.0,
      27.4,
      28.3,
      27.1,
      26.7,
      27.9,
      27.6,
      28.0,
      26.5,
      26.1,
      27.6,
      26.9]

ESCLUSE = [12, 13]
#ESCLUSE = [10, 11]
#ESCLUSE = []

c = []
sigma_stats = []
mat = []

for i in range(len(m1)):
    if i + 1 in ESCLUSE:
        continue
    c.append(CA * (ma[i] + ME) * (Tf[i] - Ta[i]) / (m1[i] * (T1[i] - Tf[i])))
    sigma_stats.append(np.sqrt((CA * (Tf[i] - Ta[i])/(m1[i] * (T1[i] - Tf[i])) * SIGMA_M)**2 +              # m_a
                       (c[-1]/m1[i] * SIGMA_M)**2 +                                                           # m_1
                       (CA * (ma[i] + ME)/(m1[i] * (T1[i] - Tf[i])) * SIGMA_T)**2 +                           # T_a
                       (c[-1]/(T1[i] - Tf[i]) * SIGMA_T)**2 +                                                 # T_1
                       (CA * (ma[i] + ME) * (T1[i] - Ta[i])/(m1[i] * (T1[i] - Tf[i])**2) * SIGMA_T)**2 +      # T_f
                       (CA * (Tf[i] - Ta[i])/(m1[i] * (T1[i] - Tf[i])) * SIGMA_ME)**2
                       ))
    mat.append(materiale[i])
    print(f"Misura {i+1} ({materiale[i]}): c = {c[-1]:.3f} ± {sigma_stats[-1]:.3f}")

# media pesata per materiale: pesi 1/sigma_stat^2, l'errore su Me è comune a tutte le misure e non si riduce
c = np.array(c)
sigma_stats = np.array(sigma_stats)
mat = np.array(mat)

for nome in C_ATTESI:
    sel = mat == nome
    w = 1/sigma_stats[sel]**2
    media = np.sum(w * c[sel]) / np.sum(w)
    s_media = 1/np.sqrt(np.sum(w))
    print(f"Media {nome}: c = {media:.3f} ± {s_media:.3f} (stat) | scarto dal valore atteso: {abs(media - C_ATTESI[nome])/s_media:.1f} sigma")

# istrogramma dell'alluminio con gaussiana sovrapposta
alum = c[mat == "alluminio"]

for bin in range(1, 10):
      
      plt.hist(alum, bins=bin, alpha=0.7, label="Alluminio")
      plt.xlabel("Calore specifico")
      plt.ylabel("Frequenza")
      plt.title("Distribuzione della calore specifico dell'alluminio")
      
      plt.legend()
      plt.savefig(f"Img/alluminumIsto_bin{bin}_no-gauss.png")
      plt.close()
      
      plt.hist(alum, bins=bin, alpha=0.7, label="Alluminio")
      plt.xlabel(r"Calore specifico $\left(\frac{cal}{g \cdot K}\right)$")
      plt.ylabel("Frequenza")
      plt.title("Distribuzione della calore specifico dell'alluminio")
      
      gauss = np.linspace(min(alum), max(alum), 100)
      media = np.mean(alum)
      sigma = np.std(alum, ddof=1)
      plt.plot(gauss, len(alum) * (max(alum) - min(alum)) / bin * (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((gauss - media)/sigma)**2), color="red", label="Gaussiana")
      
      plt.legend()
      plt.savefig(f"Img/alluminumIsto_bin{bin}.png")
      plt.close()
      #plt.show()
      
#calcolo del chi quadro per l'alluminio
sel = mat == "alluminio"
media = np.sum(c[sel]/sigma_stats[sel]**2)/np.sum(1/sigma_stats[sel]**2)
chi2 = np.sum(((c[sel] - media)/sigma_stats[sel])**2)
ndof = np.sum(sel) - 1

print(f"Chi quadro per l'alluminio: {chi2:.3f} con {ndof} gradi di libertà")