# Importiamo le librerie necessarie
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

# Funzione per il fit esponenziale del rate in funzione della pressione
def esponenziale(x, a, b):
    return a * np.exp(b * x)

# Funzione di elaborazione per ogni file CSV
def elabora_file(path_csv, base_date=np.datetime64('2007-01-01')):
    # Importiamo il file CSV
    df1 = pd.read_csv(path_csv)

    # Arrotondiamo i valori di pressione a interi
    df1['Pressure'] = df1['Pressure'].round().astype(int)

    # Creiamo la colonna 'Date' come datetime
    df1['Date'] = base_date + pd.to_timedelta(df1['#BinStart'] / 86400, unit='D')

    # Estraiamo il range di pressione per il fit
    p_min, p_max = df1['Pressure'].min(), df1['Pressure'].max()
    pressure = pd.Series(range(p_min, p_max + 1))
    mean_rate = [df1.loc[df1['Pressure'] == p, 'RateHitEvents'].mean() for p in pressure]
    mean_rate = pd.Series(mean_rate)
    
    # Filtriamo i dati validi per il fit
    mask = ~mean_rate.isna()
    xdata, ydata = pressure[mask], mean_rate[mask]
    
    # Calcoliamo i parametri del fit esponenziale
    params, _ = curve_fit(esponenziale, xdata, ydata, p0=(1, -0.01))
    _, b_fit = params

    # Selezioniamo solo i valori di pressione ragionevoli
    df_valid = df1[df1['Pressure'].between(900, 1100)].copy()

    # Calcoliamo il rate corretto rispetto a una pressione di riferimento (1000 mbar)
    df_valid['RateCorr'] = df_valid['RateHitEvents'] * np.exp(b_fit * (df_valid['Pressure'] - 1000))

    # Normalizziamo il rate corretto e lo convertiamo in percentuale differenziale
    df_valid['RateCorr Norm'] = df_valid['RateCorr'] / df_valid['RateCorr'].mean()
    df_valid['RateCorr %'] = (df_valid['RateCorr Norm'] - 1) * 100

    # Raggruppiamo i dati su intervalli orari (media per ogni ora)
    df_valid['Hour'] = df_valid['Date'].dt.floor('1H')
    df_grouped = df_valid.groupby('Hour')['RateCorr %'].mean().reset_index()

    # Limitiamo il range di valori percentuali per evitare outlier visivi
    df_grouped = df_grouped[df_grouped['RateCorr %'].between(-10, 10)]

    # Rinominiamo le colonne e aggiungiamo il nome del file come etichetta
    nome_file = os.path.basename(path_csv).split('.')[0]
    df_grouped.rename(columns={'Hour': 'Date', 'RateCorr %': 'Rate (%)'}, inplace=True)
    df_grouped['File'] = nome_file

    return df_grouped

# Funzione per creare il grafico finale
def plot_finale(dfs, data_inizio, data_fine):
    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    # Cicliamo su tutti i dataframe dei telescopi
    for df in dfs:
        # Filtriamo l’intervallo desiderato
        df = df[(df['Date'] >= data_inizio) & (df['Date'] <= data_fine)]
        label = df['File'].iloc[0]
        # Tracciamo i dati con marker a pallino
        ax.plot(df['Date'], df['Rate (%)'], label=label, marker='o', markersize=3)

    # Impostazioni del grafico
    plt.axvline(datetime(2024, 5, 10, 16, 45), color='magenta', linestyle='--', linewidth=1.5,
                label='G5 storm start\n10/5/2024 16:45 UTC')

    plt.title('Forbush Maggio 2024', fontsize=18, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('differential rate (%)')
    plt.xticks(rotation=45)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Blocco principale per elaborare tutti i file e fare il grafico
if __name__ == '__main__':
    # Indichiamo la cartella dove si trovano i file CSV
    cartella = 'dati'  

    # Lista dei file da elaborare
    file_csv = [f for f in os.listdir(cartella) if f.endswith('.csv')]

    # Elaboriamo tutti i file
    risultati = []
    for file in file_csv:
        path_completo = os.path.join(cartella, file)
        df_result = elabora_file(path_completo)
        risultati.append(df_result)

    # Intervallo temporale per il grafico finale
    data_inizio = datetime(2024, 5, 8)
    data_fine = datetime(2024, 5, 15)

    # Generiamo il grafico finale con tutti i telescopi
    plot_finale(risultati, data_inizio, data_fine)