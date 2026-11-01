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


soglia_inizio_arr = [0.6, 0.2]
soglia_fine_arr = [0.3, 0.3]

type = ["acqua", "aria"]

for i in range(1, 3):
    file = f"7/7-{i}"
    t, T = np.loadtxt(f"{file}.txt", unpack=True, skiprows=2, encoding='latin1', converters={0: comma_to_float, 1: comma_to_float})

    soglia_inizio = soglia_inizio_arr[i - 1]
    soglia_fine = soglia_fine_arr[i - 1]

    #STIMO T0 E TBAR
    n_punti_stima = 50

    T_plateau_iniziale = T[:n_punti_stima]
    T_plateau_finale = T[-n_punti_stima:]

    T0 = np.average(T_plateau_iniziale)
    Tbar = np.average(T_plateau_finale)

    #INCERTEZZA DI TIPO A (dispersione della singola lettura a temperatura costante)
    sigma_a_T0 = np.std(T_plateau_iniziale, ddof=1)
    sigma_a_Tbar = np.std(T_plateau_finale, ddof=1)
    # media in quadratura delle due stime
    sigma_a_temperatura = np.sqrt((sigma_a_T0**2 + sigma_a_Tbar**2) / 2)

    #INCERTEZZA TOTALE SULLA SINGOLA LETTURA
    sigma_temperatura = np.sqrt(sigma_a_temperatura**2 + info.sigma_b_temperatura_CI6605**2)
    
    #CERCO PUNTI DI INIZIO / FINE DEL TRANSITORIO
    indici_inizio = np.where(np.abs(T - T0) > soglia_inizio)[0]
    idx_inizio = indici_inizio[0] if len(indici_inizio) > 0 else 0
    tempo_inizio = t[idx_inizio]

    indici_fuori_regime = np.where(np.abs(T - Tbar) > soglia_fine)[0]
    if len(indici_fuori_regime) > 0:
        idx_fine = indici_fuori_regime[-1] + 1  
    else:
        idx_fine = len(t) - 1

    tempo_fine = t[idx_fine]
    
    print(f"Stima in {type[i - 1]}: \n")
    print(f"T0 stimata: {T0:.2f} | Tbar stimata: {Tbar:.2f}")
    print(f"sigma_A (T0): {sigma_a_T0:.3f} | sigma_A (Tbar): {sigma_a_Tbar:.3f} | sigma_A usata: {sigma_a_temperatura:.3f}")
    print(f"sigma_B: {info.sigma_b_temperatura_CI6605:.3f} | sigma totale: {sigma_temperatura:.3f}")
    print(f"Inizio misura a t={tempo_inizio} | Fine misura a t={tempo_fine}")

    #PLOTTO I RISULTATI
    plt.figure(figsize=(10, 6))
    plt.plot(t, T, label='Dati sperimentali', color='black')

    plt.axhline(T0, color='blue', linestyle='--', label=f'T0 = {T0:.2f}')
    plt.axhline(Tbar, color='red', linestyle='--', label=f'Tbar = {Tbar:.2f}')

    plt.axvline(tempo_inizio, color='blue', linestyle='-', label=f'Start (soglia {soglia_inizio})')
    plt.axvline(tempo_fine, color='red', linestyle='-', label=f'End (soglia {soglia_fine})')

    plt.legend()
    plt.title(f"Analisi Transitorio Termico [{file}]")
    plt.xlabel("Tempo")
    plt.ylabel("Temperatura")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{file}_analisi_transitorio.png", dpi = 300)
    plt.show()

    #PLOTTO IN FORMA LOGARITMICA

    # Filtriamo i dati
    t_per_analisi = t[idx_inizio:idx_fine]
    T_per_analisi = T[idx_inizio:idx_fine]

    x = t_per_analisi
    
    argomento_log = np.abs(Tbar - T_per_analisi)
    y = np.log(argomento_log)
    
    sigma_y = sigma_temperatura / argomento_log
            
    coefficienti, matrice_cov = np.polyfit(t_per_analisi, y, deg=1, w=1/sigma_y, cov=True)

    incertezze = np.sqrt(np.diag(matrice_cov))

    m, q = coefficienti
    errore_m, errore_q = incertezze

    print(f"Coefficiente angolare (m): {m:.4f} ± {errore_m:.4f} | Tau stimato: {-1/m:.4f}")
    print(f"Intercetta (q): {q:.4f} ± {errore_q:.4f}")

    plt.figure(figsize=(10, 6))
    
    plt.errorbar(t_per_analisi, 
                 y, 
                 yerr=sigma_y, 
                 fmt='o', 
                 color='black', 
                 markersize = 4,
                 ecolor='gray', 
                 elinewidth=1, 
                 capsize=2, 
                 label='Dati con incertezza'
                 )
    
    plt.plot(t_per_analisi, 
             m * t_per_analisi + q, 
             '-', 
             label=f'Fit lineare: y = {m:.4f}x + {q:.4f}', 
             color='orange'
             )

    y0 = np.log(np.abs(Tbar - T0))
    plt.axhline(y0, 
                color='blue', 
                linestyle='--', 
                label=f'ln(|Tbar - T0|) = {y0:.2f}'
                )

    plt.legend()
    plt.title(f"Linearizzazione per il calcolo di Tau [{file}]")
    plt.xlabel("Tempo [s]")
    plt.ylabel("ln(|Tbar - T(t)|)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{file}_linearizzazione_per_tau.png", dpi = 300)
    plt.show()

       
    print("")