import numpy as np

# ----------------------------------------------------------------------------
# PARAMETRI SPERIMENTALI
# ----------------------------------------------------------------------------
CA = 1.0          # cal/(g K), calore specifico dell'acqua
ME = 25.0         # g, massa equivalente del thermos
SIGMA_ME = 5.0    # g
SIGMA_M = np.sqrt((0.1/np.sqrt(12))**2 + (0.3)**2)     # g, risoluzione bilancia + linearity
SIGMA_T = np.sqrt((0.2/np.sqrt(12))**2 + (1)**2)  # °C, incertezza termometro + accuracy del termometro

TG = 0.0          # °C, temperatura del ghiaccio (fusione)

LAMBDA_ATTESO = 79.7  # cal/g, valore di riferimento

ma = [200.3,
      198.3]
Ta = [55.0,
      54.8]
mg = [70.4,
      80.8]
Tg = [0.4,
      0.6]
Tf = [22.4,
      21.1]

c = []
sigma_stats = []

for i in range(len(ma)):
    c.append(CA * ((ma[i] + ME) * (Ta[i] - Tf[i]) - mg[i] * (Tf[i] - Tg[i])) / mg[i])
    sigma_stats.append(np.sqrt((CA * (Tf[i] - Ta[i])/mg[i] * SIGMA_M)**2 + 
                       (CA * (ma[i] + ME) * (Ta[i] - Tf[i])/mg[i]**2 * SIGMA_M)**2 +
                       (CA * (ma[i] + ME)/mg[i] * SIGMA_T)**2 +
                       (CA * SIGMA_T)**2 + 
                       (CA * (ma[i] + ME + mg[i])/mg[i] * SIGMA_T)**2 + 
                       (CA * (- Tf[i] + Ta[i])/mg[i] * SIGMA_ME)**2
                       ))
    
for i in range(len(ma)):
    print(f"Misura {i+1}: c = {c[i]:.2f} ± {sigma_stats[i]:.2f} ")
    
sigma_stats = np.array(sigma_stats)

    
w = 1/sigma_stats**2
media = np.sum(w * c) / np.sum(w)
s_media = 1/np.sqrt(np.sum(w))

print(f"Media: c = {media:.2f} ± {s_media:.2f} | scarto dal valore atteso: {abs(media - LAMBDA_ATTESO)/s_media:.1f} sigma")
