import numpy as np
import numpy as np

# Postazione 6 -> configurazione postazioni 1-20 (DataStudio)
# Pressione:   Low Pressure Sensor   CI-6534
# Temperatura: Temperature Sensor   CI-6605
# Posizione:   Rotary Motion Sensor  CI-6538 (puleggia media, 1440 conteggi/giro)
# Massa:       Kern FCB-3K0.1

risoluzione_pressione_CI6534 = 0.005  # kPa
risoluzione_temperatura_CI6605 = 0.05  # °C

raggio_puleggia_media_CI6538 = 48/2  # mm
risoluzione_posizione_CI6538 = 2 * np.pi * raggio_puleggia_media_CI6538 / 1440  # mm

print(risoluzione_posizione_CI6538)
risoluzione_massa_FCB3K01 = 0.1 # g

accuracy_pressione_CI6534 = 0.102
accuracy_temperatura_CI6605 = 0.5
accuracy_posizione_CI6538 = 0
accuracy_massa_FCB3K01 = 0.3 # g

sigma_b_pressione_CI6534 = np.sqrt((risoluzione_pressione_CI6534 / np.sqrt(12))**2 + accuracy_pressione_CI6534**2)
sigma_b_temperatura_CI6605 = np.sqrt((risoluzione_temperatura_CI6605 / np.sqrt(12))**2 + accuracy_temperatura_CI6605**2)
sigma_b_posizione_CI6538 = np.sqrt((risoluzione_posizione_CI6538 / np.sqrt(12))**2 + accuracy_posizione_CI6538**2)
sigma_b_massa_FCB3K01 = np.sqrt((risoluzione_massa_FCB3K01 / np.sqrt(12))**2 + accuracy_massa_FCB3K01**2)