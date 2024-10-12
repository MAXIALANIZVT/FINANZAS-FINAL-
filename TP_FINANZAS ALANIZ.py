import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime

 ################ IMPORTANTE #########################################################################
 #### TRABAJO PRACTICO Programación en Python aplicada a Finanzas - Docente: Dr. Luciano Machain    ##
 #####################################################################################################
 ## Tasa libre de riesgo argentina 4,1% BONO 10 AÑOS USA + 1140 RIESGO PAIS                         ##
 #####################################################################################################
 ## Los calculos se toman diarios, en una serie de 1 año, por ende es un portafolio de corto plazo  ##
 #####################################################################################################

#  Obtener la fecha actual y la fecha de hace un año


hoy = datetime.today()
start_date = hoy.replace(year=hoy.year - 2)  # Esto se puede cambiar, pero tambien los rendimientos anualizados deben editarse.

#  Descargar cotizaciones de los activos
def descargar_datos_activos(activos, start_date=start_date, period='1d'):
    return pd.DataFrame({
        activo: yf.download(activo, start=start_date,end=None, period=period)["Adj Close"]
        for activo in activos
    })

#  Calcular rendimientos y covarianza
def calcular_rendimientos(datos):
    rendimientos_diarios = datos.pct_change().dropna()
    rendimientos_anualizados = (1 + rendimientos_diarios.mean()) ** 256 - 1
    rendimientos_acumulados = (1 + rendimientos_diarios).cumprod() - 1
    rendimiento_total = rendimientos_acumulados.iloc[-1]

    covarianza = rendimientos_diarios.cov() * 256
    correlacion = rendimientos_diarios.corr()

    return rendimientos_anualizados, covarianza, correlacion, rendimientos_acumulados, rendimiento_total


#  Descargar datos Merval
def descargar_datos_merval():
    """Descarga los datos históricos del ticker de Merval"""
    datos_merval = yf.download("^MERV", start=start_date, end=None, period="1d")["Adj Close"]
    return datos_merval

#  Calcular rendimiento y riesgo del Merval
def calcular_rendimiento_riesgo_merval(datos_merval):
    rendimientos_diarios_merval = datos_merval.pct_change().dropna()
    rendimiento_anualizado_merval = (1 + rendimientos_diarios_merval.mean()) ** 256 - 1  # 191 días de negociación
    riesgo_merval = rendimientos_diarios_merval.std() * np.sqrt(256)  # Desviación estándar anualizada

    return rendimiento_anualizado_merval, riesgo_merval
 

# Simular portafolios // 1 millone! 
def simular_portafolios(rendimientos, covarianza, n_simulaciones=100000):
    n = len(rendimientos)
    resultados = np.zeros((3, n_simulaciones))
    pesados_portafolios = np.zeros((n, n_simulaciones))

    for i in range(n_simulaciones):
        pesos = np.random.random(n)
        pesos /= pesos.sum()

        rendimiento = np.dot(pesos, rendimientos)
        riesgo = np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))

        resultados[0, i] = riesgo
        resultados[1, i] = rendimiento
        pesados_portafolios[:, i] = pesos

    return resultados, pesados_portafolios

#  Calcular el ratio de Sharpe
def calcular_ratio_sharpe(rendimiento, riesgo, tasa_libre_riesgo=0.155):
    return (rendimiento - tasa_libre_riesgo) / riesgo if riesgo != 0 else 0


#  Graficar de simulaciones de portafolio
def graficar_frontera_eficiente_con_sharpe(resultados_portafolios, pesados_portafolios):
    rendimientos, riesgos = resultados_portafolios[1], resultados_portafolios[0]
    ratios_sharpe = np.array([calcular_ratio_sharpe(rend, riesgo) for rend, riesgo in zip(rendimientos, riesgos)])

    mejor_indice = np.argmax(ratios_sharpe)
    mejor_riesgo, mejor_rendimiento = riesgos[mejor_indice], rendimientos[mejor_indice]

    # Mostrar valores en la consola
    print(f"Rendimiento esperado del portafolio: {(mejor_rendimiento*100):.2f}%")
    print(f"Riesgo del portafolio: {(mejor_riesgo*100):.2f}%")

    trace = go.Scatter(
        x=riesgos*100, y=rendimientos*100, mode='markers',
        marker=dict(color=ratios_sharpe, colorscale='Viridis', size=8, colorbar=dict(title="Ratio Sharpe")),
        name="Portafolios Simulados"
    )

    trace_mejor = go.Scatter(
        x=[mejor_riesgo*100], y=[mejor_rendimiento*100], mode='markers',
        marker=dict(color='red', size=12, symbol='star'), name="Portafolio"
    )

    trace_merval = go.Scatter(
        x=[riesgo_merval*100], y=[rendimiento_merval*100], mode='markers',
        marker=dict(color="red", size=12), name="Merval"
    )

    layout = go.Layout(
        title="Portafolios Simulados",
        xaxis=dict(title='Riesgo %'),
        yaxis=dict(title='Rendimiento Esperado %'),
        showlegend=True,
        legend=dict(
            orientation="h",  
            yanchor="bottom",  
            y=-0.15,  
            xanchor="center",  
            x=0.5  
        )
    )

    fig = go.Figure(data=[trace, trace_mejor, trace_merval], layout=layout)
    fig.show()


from scipy.optimize import minimize
import numpy as np

#  Optimizar portafolio para maximizar el ratio de Sharpe
def optimizar_portafolio_sharpe(rendimientos, covarianza, tasa_libre_riesgo=0.155):
    n = len(rendimientos)
    
    # Función para calcular el rendimiento esperado del portafolio
    def calcular_rendimiento(pesos):
        return np.sum(pesos * rendimientos)
    
    # Función para calcular el riesgo (desviación estándar) del portafolio
    def calcular_riesgo(pesos):
        return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))
    
    # Función para calcular el negativo del ratio de Sharpe (queremos maximizar, pero el optimizador minimiza)
    def objetivo(pesos):
        rendimiento_portafolio = calcular_rendimiento(pesos)
        riesgo_portafolio = calcular_riesgo(pesos)
        sharpe_ratio = (rendimiento_portafolio - tasa_libre_riesgo) / riesgo_portafolio
        return -sharpe_ratio  # Negamos el ratio de Sharpe para maximizarlo
    
    # Restricción: la suma de los pesos debe ser igual a 1
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Límites: cada peso debe estar entre 0 y 1
    limites = [(0, 1)] * n
    
    # Pesos iniciales: distribución uniforme
    pesos_iniciales = np.ones(n) / n
    
    # Realizar la optimización usando SLSQP
    resultado = minimize(objetivo, pesos_iniciales, method='SLSQP', bounds=limites, constraints=restricciones)
    
    # Resultados óptimos
    pesos_optimos = resultado.x
    rendimiento_optimo = calcular_rendimiento(pesos_optimos) * 100
    riesgo_optimo = calcular_riesgo(pesos_optimos) * 100
    sharpe_optimo = (rendimiento_optimo - tasa_libre_riesgo * 100) / riesgo_optimo

    return pesos_optimos, rendimiento_optimo, riesgo_optimo, sharpe_optimo

#  Graficar mapa de calor de la correlación
def graficar_mapa_calor(correlacion):

    # Eliminar el sufijo ".BA" de los índices y columnas
    correlacion.columns = correlacion.columns.str.replace('.BA', '')
    correlacion.index = correlacion.index.str.replace('.BA', '')

    # Crear el gráfico con Plotly
    fig = px.imshow(correlacion, text_auto=False, aspect="equal", color_continuous_scale="Blugrn", origin="Upper")
    fig.update_layout(title_text='Mapa de Calor de la Correlación', title_x=0.5)
    fig.show()



#  Graficar gráfico circular con pesos óptimos
def graficar_grafico_circular(pesos_optimos, activos):
    pesos_filtrados = pesos_optimos[pesos_optimos > 0.01] * 100
    activos_filtrados = [activo.replace('.BA', '') for i, activo in enumerate(activos) if pesos_optimos[i] > 0.01]

    if len(activos_filtrados) == 0:
        print("No hay activos con un peso mayor al 1% para graficar.")
        return

    fig = go.Figure(data=[go.Pie(labels=activos_filtrados, values=pesos_filtrados, textinfo='percent+label', hole=0.3)])
    fig.update_layout(title_text="Distribución de activos de la cartera")
    fig.show()


# Ejecución del script
if __name__ == "__main__":
    activos = ["YPFD.BA", "GGAL.BA", "BMA.BA", "TECO2.BA", "CRES.BA", 
               "EDN.BA", "BBAR.BA", "ALUA.BA", "COME.BA", "PAMP.BA", 
               "METR.BA", "TGSU2.BA", "CEPU.BA", "LOMA.BA", "SUPV.BA", 
               "BYMA.BA", "VALO.BA", "TGNO4.BA", "TRAN.BA", "HAVA.BA", 
               "TXAR.BA", "CELU.BA", "BHIP.BA", "LEDE.BA", "BOLT.BA", 
               "AUSO.BA", "AGRO.BA", "MORI.BA", "HARG.BA", "MOLI.BA", 
               "BPAT.BA", "SEMI.BA", "CVH.BA", "LONG.BA", "GCLA.BA", 
               "CARC.BA", "OEST.BA", "CECO2.BA", "CADO.BA", "MIRG.BA"
               ]
    


    # Descargar datos
    datos_activos = descargar_datos_activos(activos)
    
    rendimientos_anualizados, covarianza ,correlacion,rendimientos_acumulados, rendimiento_total= calcular_rendimientos(datos_activos)
    # Convertir 'rendimiento_total' a DataFrame para eliminar metadatos
    rendimiento_df = pd.DataFrame(rendimiento_total*100)

    # Imprimir solo los datos sin la línea 'Name:...'
    print("Rendimientos: \n", rendimiento_df.to_string(header=False))

    
    # Descargar y calcular Merval
    datos_merval = descargar_datos_merval()
    rendimiento_merval, riesgo_merval = calcular_rendimiento_riesgo_merval(datos_merval)
    print("Rendimiento Esperado Indice Merval:", (rendimiento_merval * 100).round(2), "%")
    print("Riesgo del Indice Merval:", (riesgo_merval*100).round(2), "%")
    
    ratio_sharpe_merval = calcular_ratio_sharpe(rendimiento_merval, riesgo_merval, tasa_libre_riesgo=0.155)
  
    # Simular portafolios y graficar frontera eficiente
    resultados_portafolios, pesados_portafolios = simular_portafolios(rendimientos_anualizados.values, covarianza.values)

    # Optimizar portafolio y mostrar resultados
    pesos_optimos, rendimiento_optimo, riesgo_optimo, sharpe_optimo = optimizar_portafolio_sharpe(rendimientos_anualizados.values, covarianza.values)

    # Graficar mejor portafolio según ratio Sharpe
    graficar_frontera_eficiente_con_sharpe(resultados_portafolios, pesados_portafolios)

    # Graficar mapa de calor de la correlación
    graficar_mapa_calor(correlacion)

    # Graficar gráfico circular con los pesos óptimos
    graficar_grafico_circular(pesos_optimos, activos)

    input("Presiona cualquier tecla para salir...")

