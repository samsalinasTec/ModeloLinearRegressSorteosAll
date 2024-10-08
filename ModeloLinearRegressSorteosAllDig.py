import numpy as np
import math as mt
import importlib
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import seaborn as sns
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from ClassFunctions import SorteosTecLinealRegress, DNASColumn, BQLoad
from google.cloud import bigquery
from google.oauth2 import service_account

dfBoletosDig=pd.read_csv("/home/stadmin/AutomatizacionScripts/DashboardNacionalAnalytics/TablasDMyFC/FCVentas_digital.csv")
dfInfo=pd.read_csv("/home/stadmin/AutomatizacionScripts/DashboardNacionalAnalytics/dataframesPreparacion/dfHistoricoSorteos.csv")

dfBoletos=dfBoletosDig[["ID_SORTEO","ID_SORTEO_DIA","FECHAREGISTRO","CANTIDAD_BOLETOS","CANAL_DIG"]].rename({"CANAL_DIG":"CANAL"},axis=1)
dfBoletos["FECHAREGISTRO"]=pd.to_datetime(dfBoletos.loc[:,"FECHAREGISTRO"],format='%Y-%m-%d')
dfInfo["FECHA_CIERRE"]=pd.to_datetime(dfInfo.loc[:,"FECHA_CIERRE"],format="%Y-%m-%d")

dfBoletos=pd.merge(dfBoletos,dfInfo[["ID_SORTEO","NOMBRE","FECHA_CIERRE","PRECIO_UNITARIO"]],on="ID_SORTEO",how="left")
dfBoletos["DNAS"]=DNASColumn(dfBoletos["FECHA_CIERRE"],dfBoletos["FECHAREGISTRO"])
dfBoletos= dfBoletos.sort_values("FECHAREGISTRO")
dfBoletos['CANTIDAD_BOLETOS_MEMBRESIAS'] = dfBoletos.apply(lambda row: row['CANTIDAD_BOLETOS'] if row['CANAL'] == 'Membresias' else 0, axis=1)
dfBoletos['CANTIDAD_BOLETOS_SIN_MEMBRE'] = dfBoletos.apply(lambda row: row['CANTIDAD_BOLETOS'] - row['CANTIDAD_BOLETOS_MEMBRESIAS'], axis=1)
dfBoletos=dfBoletos.drop("CANAL",axis=1)

dfBoletosDlasSin0=dfBoletos.groupby(["NOMBRE","DNAS"]).agg(CANTIDAD_BOLETOS=("CANTIDAD_BOLETOS","sum"),CANTIDAD_BOLETOS_SIN_MEMBRE=("CANTIDAD_BOLETOS_SIN_MEMBRE","sum"),CANTIDAD_BOLETOS_MEMBRESIAS=("CANTIDAD_BOLETOS_MEMBRESIAS","sum")).reset_index()
sorteos=dfBoletos["NOMBRE"].unique()
for sorteoVariable in sorteos:

    boletosDiasNegativos= dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] <= 0)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS_SIN_MEMBRE'].sum()
    dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] == 1)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS_SIN_MEMBRE'] += boletosDiasNegativos

    boletosDiasNegativos= dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] <= 0)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS_MEMBRESIAS'].sum()
    dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] == 1)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS_MEMBRESIAS'] += boletosDiasNegativos

    boletosDiasNegativos= dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] <= 0)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS'].sum()
    dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] == 1)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS'] += boletosDiasNegativos

    # Eliminar las filas con días negativos
dfBoletosDlasSin0 = dfBoletosDlasSin0[dfBoletosDlasSin0['DNAS'] > 0]
dfBoletosDlasSin0=dfBoletosDlasSin0.sort_values("DNAS",ascending=False)
dfBoletosDlasSin0["BOLETOS_ACUMULADOS_SIN_MEMBRE"] = dfBoletosDlasSin0.groupby("NOMBRE")["CANTIDAD_BOLETOS_SIN_MEMBRE"].cumsum()
dfBoletosDlasSin0["BOLETOS_ACUMULADOS_CON_MEMBRE"] = dfBoletosDlasSin0.groupby("NOMBRE")["CANTIDAD_BOLETOS"].cumsum()


dfBoletosEscalado=pd.merge(dfBoletosDlasSin0,dfInfo[["NOMBRE","EMISION","ID_SORTEO"]],on="NOMBRE",how="left")
dfBoletosEscalado["PORCENTAJE_DE_AVANCE_SIN_MEMBRE"]=dfBoletosEscalado["BOLETOS_ACUMULADOS_SIN_MEMBRE"]/dfBoletosEscalado["EMISION"]
dfBoletosEscalado["PORCENTAJE_DE_AVANCE_CON_MEMBRE"]=dfBoletosEscalado["BOLETOS_ACUMULADOS_CON_MEMBRE"]/dfBoletosEscalado["EMISION"]


max_days = dfBoletosEscalado.groupby('NOMBRE')['DNAS'].transform('max')-1
dfBoletosEscalado["PORCENTAJE_DNAS"]=(max_days-(dfBoletosEscalado["DNAS"]-1))/max_days
dfBoletosEscalado=dfBoletosEscalado.sort_values(["NOMBRE","DNAS"],ascending=False)


dfPrediccionesTST217=SorteosTecLinealRegress("Sorteo Tradicional 217",270_000,pd.to_datetime("21/12/2024",dayfirst=True),["Sorteo Tradicional 211","Sorteo Tradicional 213","Sorteo Tradicional 215","Sorteo Tradicional 217"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE_SIN_MEMBRE",dfBoletosEscalado).predict()
dfPrediccionesAVT29=SorteosTecLinealRegress("Sorteo AventuraT 29",80_000,pd.to_datetime("18/01/2025",dayfirst=True),["Sorteo AventuraT 23","Sorteo AventuraT 25","Sorteo AventuraT 27","Sorteo AventuraT 29"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE_SIN_MEMBRE",dfBoletosEscalado).predict()
dfPrediccionesSOE47=SorteosTecLinealRegress("Sorteo Educativo 47",390_000,pd.to_datetime("19/10/2024",dayfirst=True),["Sorteo Educativo 41","Sorteo Educativo 43","Sorteo Educativo 45","Sorteo Educativo 47"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE_SIN_MEMBRE",dfBoletosEscalado).predict()
dfPrediccionesSMS30=SorteosTecLinealRegress("Sorteo Mi Sueño 30",260_000,pd.to_datetime("23/11/2024",dayfirst=True),["Sorteo Mi Sueño 25","Sorteo Mi Sueño 27","Sorteo Mi Sueño 28","Sorteo Mi Sueño 29"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE_SIN_MEMBRE",dfBoletosEscalado).predict()
dfPrediccionesLQ11=SorteosTecLinealRegress("LQ 11",80_000,pd.to_datetime("28/05/2025",dayfirst=True),["LQ 8","LQ 9","LQ 10","LQ 11"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE_SIN_MEMBRE",dfBoletosEscalado).predict()


listDataframes = [v for k, v in globals().items() if k.startswith('dfPredicciones')]
dfPrediccionesTodos = pd.concat(listDataframes, ignore_index=True)
dfPrediccionesTodos.to_csv("/home/stadmin/AutomatizacionScripts/ModeloLinearRegressSorteosAll/dfPrediccionesTodosDigital.csv",header=True,index=False)

jobConfigFCPredicciones_nacional = bigquery.LoadJobConfig(
schema = [
    bigquery.SchemaField("ID_SORTEO","INTEGER"),
    bigquery.SchemaField("ID_SORTEO_DIA","INTEGER"),
    bigquery.SchemaField("SORTEO","STRING"),
    bigquery.SchemaField("DNAS","INTEGER"),
    bigquery.SchemaField("TALONES_ESTIMADOS","FLOAT"),
    bigquery.SchemaField("TALONES_DIARIOS_ESTIMADOS","FLOAT"),
    bigquery.SchemaField("FECHA_MAPEADA","DATE")
])

BQLoad("/home/stadmin/AutomatizacionScripts/ModeloLinearRegressSorteosAll/dfPrediccionesTodosDigital.csv","FECHA_MAPEADA",jobConfigFCPredicciones_nacional,"FCPredicciones_digital","sorteostec-analytics360.PruebasDashboardNacional")