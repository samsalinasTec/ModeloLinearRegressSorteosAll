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

dfBoletosDig=dfBoletosDig.loc[dfBoletosDig["CANAL_DIG"]!="Membresias"]

dfBoletosDig=dfBoletosDig[["ID_SORTEO","ID_SORTEO_DIA","FECHAREGISTRO","CANTIDAD_BOLETOS"]]
dfBoletosDig["FECHAREGISTRO"]=pd.to_datetime(dfBoletosDig.loc[:,"FECHAREGISTRO"],format='%Y-%m-%d')
dfInfo["FECHA_CIERRE"]=pd.to_datetime(dfInfo.loc[:,"FECHA_CIERRE"],format="%Y-%m-%d")

dfBoletosDig=pd.merge(dfBoletosDig,dfInfo[["ID_SORTEO","NOMBRE","FECHA_CIERRE"]],on="ID_SORTEO",how="left")
dfBoletosDig["DNAS"]=DNASColumn(dfBoletosDig["FECHA_CIERRE"],dfBoletosDig["FECHAREGISTRO"])
dfBoletosDig= dfBoletosDig.sort_values("FECHAREGISTRO")

dfBoletosDlasSin0=dfBoletosDig.groupby(["NOMBRE","DNAS"])["CANTIDAD_BOLETOS"].sum().reset_index()
sorteos=dfBoletosDig["NOMBRE"].unique()
for sorteoVariable in sorteos:

    boletosDiasNegativos= dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] <= 0)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS'].sum()
    dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] == 1)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS'] += boletosDiasNegativos

    # Eliminar las filas con días negativos
dfBoletosDlasSin0 = dfBoletosDlasSin0[dfBoletosDlasSin0['DNAS'] > 0]
dfBoletosDlasSin0=dfBoletosDlasSin0.sort_values("DNAS",ascending=False)
dfBoletosDlasSin0["BOLETOS_ACUMULADOS"] = dfBoletosDlasSin0.groupby("NOMBRE")["CANTIDAD_BOLETOS"].cumsum()
combinacionSorteos=list(itertools.combinations(sorteos, 3))

dfBoletosEscalado=pd.merge(dfBoletosDlasSin0,dfInfo[["NOMBRE","EMISION","ID_SORTEO"]],on="NOMBRE",how="left")
dfBoletosEscalado["PORCENTAJE_DE_AVANCE"]=dfBoletosEscalado["BOLETOS_ACUMULADOS"]/dfBoletosEscalado["EMISION"]

max_days = dfBoletosEscalado.groupby('NOMBRE')['DNAS'].transform('max')-1
dfBoletosEscalado["PORCENTAJE_DNAS"]=(max_days-(dfBoletosEscalado["DNAS"]-1))/max_days
dfBoletosEscalado=dfBoletosEscalado.sort_values(["NOMBRE","DNAS"],ascending=False)


dfPrediccionesDDXV9=SorteosTecLinealRegress("DINERO DE X VIDA 9",67_000,pd.to_datetime("21/09/2024",dayfirst=True),["DINERO DE X VIDA 5","DINERO DE X VIDA 7","DINERO DE X VIDA 9"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE",dfBoletosEscalado).predict()
dfPrediccionesTST217=SorteosTecLinealRegress("Sorteo Tradicional 217",50_000,pd.to_datetime("21/12/2024",dayfirst=True),["Sorteo Tradicional 211","Sorteo Tradicional 213","Sorteo Tradicional 215","Sorteo Tradicional 217"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE",dfBoletosEscalado).predict()
dfPrediccionesAVT29=SorteosTecLinealRegress("Sorteo AventuraT 29",35_000,pd.to_datetime("18/01/2025",dayfirst=True),["Sorteo AventuraT 23","Sorteo AventuraT 25","Sorteo AventuraT 27","Sorteo AventuraT 29"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE",dfBoletosEscalado).predict()
dfPrediccionesSOE47=SorteosTecLinealRegress("Sorteo Educativo 47",48_500,pd.to_datetime("19/10/2024",dayfirst=True),["Sorteo Educativo 41","Sorteo Educativo 43","Sorteo Educativo 45","Sorteo Educativo 47"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE",dfBoletosEscalado).predict()
dfPrediccionesSMS30=SorteosTecLinealRegress("Sorteo Mi Sueño 30",87_000,pd.to_datetime("23/11/2024",dayfirst=True),["Sorteo Mi Sueño 25","Sorteo Mi Sueño 27","Sorteo Mi Sueño 28","Sorteo Mi Sueño 29"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE",dfBoletosEscalado).predict()
dfPrediccionesDDXV10=SorteosTecLinealRegress("DINERO DE X VIDA 10",60_000,pd.to_datetime("22/02/2025",dayfirst=True),["DINERO DE X VIDA 6","DINERO DE X VIDA 8","DINERO DE X VIDA 10"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE",dfBoletosEscalado)
dfPrediccionesLQ11=SorteosTecLinealRegress("LQ 11",80_000,pd.to_datetime("28/05/2025",dayfirst=True),["LQ 8","LQ 9","LQ 10","LQ 11"],"PORCENTAJE_DNAS","PORCENTAJE_DE_AVANCE",dfBoletosEscalado)


listDataframes = [v for k, v in globals().items() if k.startswith('dfPredicciones')]
dfPrediccionesTodos = pd.concat(listDataframes, ignore_index=True)
dfPrediccionesTodos.to_csv("/home/stadmin/AutomatizacionScripts/ModeloLinearRegressSorteosAll/dfPrediccionesTodosDigital.csv",header=True,index=False)

jobConfigFCPredicciones_nacional = bigquery.LoadJobConfig(
schema = [
    bigquery.SchemaField("ID_SORTEO","INTEGER"),
    bigquery.SchemaField("ID_SORTEO_DIA","INTEGER"),
    bigquery.SchemaField("SORTEO","STRING"),
    bigquery.SchemaField("DNAS","INTEGER"),
    bigquery.SchemaField("PORCENTAJE_DNAS", "FLOAT"),
    bigquery.SchemaField("AVANCE_ESTIMADO","FLOAT"),
    bigquery.SchemaField("AVANCE_ESTIMADO_DIARIO","FLOAT"),
    bigquery.SchemaField("TALONES_ESTIMADOS","FLOAT"),
    bigquery.SchemaField("TALONES_DIARIOS_ESTIMADOS","FLOAT"),
    bigquery.SchemaField("FECHA_MAPEADA","DATE")
])

BQLoad("/home/stadmin/AutomatizacionScripts/ModeloLinearRegressSorteosAll/dfPrediccionesTodosDigital.csv","FECHA_MAPEADA",jobConfigFCPredicciones_nacional,"FCPredicciones_digital","sorteostec-analytics360.PruebasDashboardNacional")