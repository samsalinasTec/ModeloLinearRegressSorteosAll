import numpy as np
import math as mt
import pandas as pd
import oracledb as cx_Oracle
from getpass import getpass
from google.cloud import bigquery
from google.oauth2 import service_account
import logging


logging.basicConfig(filename='/home/stadmin/AutomatizacionScripts/ModeloLinearRegressSorteosAll/SQLTablasLog.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

logging.info("Inicia la descarga")

USUARIO = "SSALINAS"
PASSWORD = "Democrates1998_"
cx_Oracle.init_oracle_client()
dsn_tns = cx_Oracle.makedsn('10.4.39.27', '1521', service_name=r'ORAPRO.MTY.ITESM.MX')

idSorteo=[1,4,5,38,30,3,41,39]
numeroJuego=[212,23,23,1,3,42,1,1]


dfInfoSorteo=pd.DataFrame()
for id, numero in zip(idSorteo,numeroJuego):

        conn = cx_Oracle.connect(user=USUARIO, password=PASSWORD, dsn=dsn_tns)

        query="""SELECT sr.id_producto, sr.numero_juego, sr.nombre, sr.emision, sr.precio_unitario ,sr.fecha_inicio, sr.fecha_celebracion
                FROM   MASTERCAT.Vw_St_Sorteo_Rifa sr
                WHERE  id_producto={p} and NUMERO_JUEGO >={ed}
                """.format(p=int(id),ed=int(numero))

        dfInfoSorteoAlone = pd.read_sql_query(query, conn)
        
        conn.close()
        
        dfInfoSorteo=pd.concat([dfInfoSorteo,dfInfoSorteoAlone])
dfInfoSorteo.to_csv("/home/stadmin/AutomatizacionScripts/ModeloLinearRegressSorteosAll/dfInfoSorteoAll.csv",header=True,index=False)



dfVentaFisiAll=pd.DataFrame()
for id, numero in zip(idSorteo,numeroJuego):
        conn = cx_Oracle.connect(user=USUARIO, password=PASSWORD, dsn=dsn_tns)
        query="""WITH Sorteo_ AS (
                SELECT sr.id_juego, sr.id_producto, sr.numero_juego, sr.numero_juego_soe, sr.nombre, sr.precio_unitario
                FROM   MASTERCAT.Vw_St_Sorteo_Rifa sr
                WHERE  id_producto={p} and NUMERO_JUEGO >={ed}
                ),
                
                Cliente_Con_Boleto AS (
                SELECT  s.nombre,b.IDMEDIOVENTA,c.FECHAREGISTRO,  COUNT(b.numBoleto) AS cantidad_boletos, COUNT(b.numBoleto) * s.precio_unitario AS ingreso_por_boletos
                FROM   Sorteo_ s 
                INNER JOIN milenio.Boleto b ON s.numero_juego_soe = b.numSorteo
                LEFT JOIN milenio.recibotalondetalle c on c.numboleto=b.numboleto and c.numsorteoaplica=b.numsorteo
                
                WHERE b.idEstatusTal in (1,3,2,6)
                GROUP BY  s.nombre, s.precio_unitario, b.IDMEDIOVENTA,c.FECHAREGISTRO

                UNION ALL
                
                SELECT  s.nombre,b.IDMEDIOVENTA,c.FECHAREGISTRO, COUNT(b.numBoleto) AS cantidad_boletos, COUNT(b.numBoleto) * s.precio_unitario AS ingreso_por_boletos
                FROM   Sorteo_ s 
                INNER JOIN hermes.Boleto b ON s.numero_juego = b.numSorteo
                LEFT JOIN hermes.recibotalondetalle c on c.numboleto=b.numboleto and c.numsorteoaplica=b.numsorteo

                
                WHERE b.idEstatusTal in (1,3,2,6) 
                GROUP BY  s.nombre, s.precio_unitario, b.IDMEDIOVENTA,c.FECHAREGISTRO
                )
        select*from cliente_Con_Boleto 
                
                """.format(p=int(id),ed=int(numero))

        dfVentaFisiAllAlone= pd.read_sql_query(query, conn)
        conn.close()
        
        dfVentaFisiAll=pd.concat([dfVentaFisiAllAlone,dfVentaFisiAll])
dfVentaFisiAll.to_csv('/home/stadmin/AutomatizacionScripts/ModeloLinearRegressSorteosAll/dfVentaFisiAll.csv', header=True, index=False)



# Credenciales para las consultas a BigQuery

KEY_FILE_LOCATION = "/home/stadmin/AutomatizacionScripts/CargaBigQuery/probable-symbol-370406-a4b54345eef3.json"

credentials = service_account.Credentials.from_service_account_file(KEY_FILE_LOCATION)

client = bigquery.Client(credentials=credentials, project=credentials.project_id)

sql = """
SELECT
  type_sorteo as ID_PRODUCTO,
  num_sorteo NUMERO_JUEGO,
  sorteo NOMBRE,
  SUM(monto_ticket) as CANTIDAD_BOLETOS_TODO,
  SUM(monto_ticket * ticket_precio) as INGRESO_TODO,
    SUM(CASE
      WHEN canal LIKE 'Membresias' AND id_oficina <> 34 THEN monto_ticket
  END
    ) BOLETOS_MEMBRESIA_FISICO,
  DATE(DT_COMPRA) as FECHAREGISTRO
FROM
  `probable-symbol-370406.oficina_virtual.compra`

  WHERE
  LEFT(tag,3) in ("EPE",'TST',"DXV","DXVL","SMS","SCH","SOE","AVT")
  or LEFT(tag,6) in ("SDXVDL")
GROUP BY
   type_sorteo, num_sorteo,sorteo, DATE(DT_COMPRA)
"""

dfTalonesDig = client.query(sql).to_dataframe()

dfTalonesDig.to_csv("/home/stadmin/AutomatizacionScripts/ModeloLinearRegressSorteosAll/dfVentaDigConMem.csv",header=True,index=False)

logging.info("Finaliza la descarga")
