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
from google.cloud import bigquery
from google.oauth2 import service_account


def DNASColumn(columnaFechaMayor: pd.Series = None, columnaFechaMenor: pd.Series = None):
    if columnaFechaMayor is not None and columnaFechaMenor is not None:
        return (columnaFechaMayor - columnaFechaMenor).dt.days
    else:
        raise ValueError("Las columnas no han sido establecidas")

def BQLoad(fileDirectory,ColumnDateName,schema,tableName,bqDirectory):
    file=pd.read_csv(fileDirectory)
    file[ColumnDateName]=pd.to_datetime(file.loc[:,ColumnDateName])
    KEY_FILE_LOCATION = "/home/stadmin/AutomatizacionScripts/DashboardNacionalAnalytics/TablasDMyFC/sorteostec-analytics360-71569ac1fe19.json"
    credentials = service_account.Credentials.from_service_account_file(KEY_FILE_LOCATION)
    client = bigquery.Client(credentials=credentials,project=credentials.project_id)
    client.delete_table("sorteostec-analytics360.PruebasDashboardNacional"+"."+tableName)
    client.load_table_from_dataframe(file,bqDirectory+"."+tableName, job_config=schema)





class SorteosTecLinealRegress:
    def __init__(self,nombreSorteo,emision,fechaCelebra,sorteosEntrenamiento,X_,y_,data):

        self.nombreSorteo=nombreSorteo
        self.emision=emision
        self.sorteosEntrenamiento=sorteosEntrenamiento
        self.fechaCelebra=fechaCelebra
        self.X_=X_
        self.y_=y_
        self.data=data
    
    def predict(self): 
         
        resultados=[]

        if (self.data["NOMBRE"]==self.nombreSorteo).any():
            dfEntrena=self.data.drop(self.data.loc[self.data["NOMBRE"]==self.nombreSorteo].last_valid_index())
        else:
            dfEntrena=self.data
        self.X=pd.array(dfEntrena.loc[dfEntrena["NOMBRE"].isin(self.sorteosEntrenamiento),self.X_])
        self.y=dfEntrena.loc[dfEntrena["NOMBRE"].isin(self.sorteosEntrenamiento),self.y_]

        for j in range(1,50):
            for i in range(1,50):    
                X_train,X_test,y_train,y_test=train_test_split(self.X.reshape(-1,1),self.y,test_size=0.2,random_state=j)
                steps = [
                    ('poly', PolynomialFeatures(degree=i)),
                    ('linear', LinearRegression())
                ]
                LinearRegressionPipeline=Pipeline(steps=steps)
                LinearRegressionPipeline.fit(X_train,y_train)

                y_pred=LinearRegressionPipeline.predict(X_train)
                y_predTest=LinearRegressionPipeline.predict(X_test)

                mse = r2_score(y_test, y_predTest)

                resultados.append((j, i, mse))
                
        resultados.sort(key=lambda x: x[2], reverse=True)
        mejoresMSE = resultados[:3]
        
        X_train,X_test,y_train,y_test=train_test_split(self.X.reshape(-1,1),self.y,test_size=1,random_state=mejoresMSE[0][0])
        steps = [
            ('poly', PolynomialFeatures(degree=mejoresMSE[0][1])),
            ('linear', LinearRegression())
        ]

        LinearRegressionPipeline=Pipeline(steps=steps)
        LinearRegressionPipeline.fit(X_train,y_train)

        maxDNAS=int(dfEntrena.loc[dfEntrena["NOMBRE"]==self.nombreSorteo,"DNAS"].max())
        DNAS=range(maxDNAS,1,-1)
        self.X_toPredict=np.linspace(0,1,maxDNAS)
        self.y_predict=LinearRegressionPipeline.predict(self.X_toPredict.reshape(-1,1))

        DNASColumn=range(maxDNAS,0,-1)
        IDColumn=self.data.loc[self.data["NOMBRE"]==self.nombreSorteo,"ID_SORTEO"].max()
        PorcentDNASColumn=np.linspace(0,1,maxDNAS)
        AvanceEstimColumn= self.y_predict
        TalonesEstimColumn= self.y_predict

        PrediccionesDict={"ID_SORTEO":IDColumn,"SORTEO":self.nombreSorteo,"DNAS":DNASColumn,"PORCENTAJE_DNAS":PorcentDNASColumn,"AVANCE_ESTIMADO":AvanceEstimColumn,"TALONES_ESTIMADOS":TalonesEstimColumn*self.emision}
        dfPredicciones=pd.DataFrame(PrediccionesDict)

        

        dfPredicciones.loc[dfPredicciones["DNAS"]<=2,"TALONES_ESTIMADOS"]+=0
        dfPredicciones['TALONES_DIARIOS_ESTIMADOS'] = dfPredicciones['TALONES_ESTIMADOS'].diff()
        dfPredicciones.iloc[0,6]=dfPredicciones["TALONES_ESTIMADOS"][0]

        dfPredicciones['AVANCE_ESTIMADO_DIARIO'] = dfPredicciones['AVANCE_ESTIMADO'].diff()
        dfPredicciones.iloc[0,7]=dfPredicciones["AVANCE_ESTIMADO"][0]

        
        fechaInicio = self.fechaCelebra - timedelta(days=int(maxDNAS))
        fechaFin = self.fechaCelebra

        # Crear el rango de fechas
        rangoFechas = pd.date_range(start=fechaInicio, end=fechaFin)
        
        dfPredicciones["FECHA_MAPEADA"]=rangoFechas[1:]

        dfPredicciones["FECHAAPOYO"]=(dfPredicciones['FECHA_MAPEADA']- pd.Timestamp('1899-12-30')).dt.days
        dfPredicciones["ID_SORTEO_DIA"]=(dfPredicciones["FECHAAPOYO"].astype(str)+dfPredicciones["ID_SORTEO"].astype(str)).astype(np.int64)
        dfPredicciones=dfPredicciones.drop("FECHAAPOYO",axis=1)
        return dfPredicciones