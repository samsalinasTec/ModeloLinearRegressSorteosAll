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
from ClassFunctions import SorteosTecLinealRegress, DNASColumn, BQLoad, SorteosTecLRWM
from google.cloud import bigquery
from google.oauth2 import service_account

dfBoletosFisi=pd.read_csv("/home/stadmin/AutomatizacionScripts/DashboardNacionalAnalytics/TablasDMyFC/FCVentas_fisico.csv")
dfBoletosDig=pd.read_csv("/home/stadmin/AutomatizacionScripts/DashboardNacionalAnalytics/TablasDMyFC/FCVentas_digital.csv")
dfInfo=pd.read_csv("/home/stadmin/AutomatizacionScripts/DashboardNacionalAnalytics/dataframesPreparacion/dfHistoricoSorteos.csv")

dfBoletosDig=dfBoletosDig[["ID_SORTEO","ID_SORTEO_DIA","FECHAREGISTRO","CANTIDAD_BOLETOS","CANAL_DIG"]].rename({"CANAL_DIG":"CANAL"},axis=1)
dfBoletosFisi=dfBoletosFisi[["ID_SORTEO","ID_SORTEO_DIA","FECHAREGISTRO","CANTIDAD_BOLETOS","CANAL_TRADICIONAL"]].rename({"CANAL_TRADICIONAL":"CANAL"},axis=1)
dfBoletos=pd.concat([dfBoletosFisi,dfBoletosDig])

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


dfPrediccionesTST218=SorteosTecLRWM("Sorteo Tradicional 218",250_000,pd.to_datetime("21/06/2025",dayfirst=True),["Sorteo Tradicional 210","Sorteo Tradicional 212","Sorteo Tradicional 214","Sorteo Tradicional 216","Sorteo Tradicional 218"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE_SIN_MEMBRE",dfBoletosEscalado).predict()
dfPrediccionesSOE47=SorteosTecLRWM("Sorteo Educativo 48",390_000,pd.to_datetime("24/05/2025",dayfirst=True),["Sorteo Educativo 42","Sorteo Educativo 44","Sorteo Educativo 46","Sorteo Educativo 48"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE_SIN_MEMBRE",dfBoletosEscalado).predict()
dfPrediccionesDDXV10=SorteosTecLRWM("DINERO DE X VIDA 10",290_000,pd.to_datetime("22/02/2025",dayfirst=True),["DINERO DE X VIDA 6","DINERO DE X VIDA 8","DINERO DE X VIDA 10"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE_SIN_MEMBRE",dfBoletosEscalado).predict()
dfPrediccionesLQ11=SorteosTecLRWM("LQ 11",80_000,pd.to_datetime("28/05/2025",dayfirst=True),["LQ 8","LQ 9","LQ 10","LQ 11"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE_SIN_MEMBRE",dfBoletosEscalado).predict()
dfPrediccionesSMS31=SorteosTecLRWM("Sorteo Mi Sueño 31",270_000,pd.to_datetime("22/03/2025",dayfirst=True),["Sorteo Mi Sueño 25","Sorteo Mi Sueño 29","Sorteo Mi Sueño 28","Sorteo Mi Sueño 31"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE_SIN_MEMBRE",dfBoletosEscalado).predict()
dfPrediccionesAVT30=SorteosTecLRWM("Sorteo AventuraT 30",80_000,pd.to_datetime("12/04/2025",dayfirst=True),["Sorteo AventuraT 24","Sorteo AventuraT 26","Sorteo AventuraT 28","Sorteo AventuraT 30"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE_SIN_MEMBRE",dfBoletosEscalado).predict()



listDataframes = [v for k, v in globals().items() if k.startswith('dfPredicciones')]
dfPrediccionesTodos = pd.concat(listDataframes, ignore_index=True)
dfPrediccionesTodos.to_csv("/home/stadmin/AutomatizacionScripts/ModeloLinearRegressSorteosAll/dfPrediccionesTodosNacional.csv",header=True,index=False)

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

BQLoad("/home/stadmin/AutomatizacionScripts/ModeloLinearRegressSorteosAll/dfPrediccionesTodosNacional.csv","FECHA_MAPEADA",jobConfigFCPredicciones_nacional,"FCPredicciones_nacional","sorteostec-analytics360.PruebasDashboardNacional")