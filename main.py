#Este programa muestra la capacidad de detección de enfermedades en las plantas
#de cacao usando los modelos entrenados ganadores  para la evaluación
#si una imagen es cacao o no y si es anómalo o no 

#Se importa las librerias utilizadas, numpy, Opencv y Keras(la cual carga 
#Theano o Tensorflow)
import numpy as np
from keras import models
import cv2
import os
#Esta funcion se encarga de obtener el mapa de color HSL de una imagen que se 
#entregue
def getmapahsl(img):
    hsl = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    mapahsl = cv2.calcHist( [hsl], [0, 2], None, [45, 64], [0, 180, 0, 256] )
    mapahsl = mapahsl/np.linalg.norm(mapahsl)
    return mapahsl

#Esta función usa el primer modelo para hallar si la región a analizar es cacao
#o no
def hay_cacao(img):
    rdi=cv2.resize(img,(112,112),interpolation=cv2.INTER_CUBIC)
    rdi_data=np.expand_dims(rdi.transpose(2,0,1)/255,axis=0)
    prediccion=modelo_deteccion.predict(rdi_data)
    return prediccion
  
#Esta función usa el segundo modelo para hallar si la región a analizar es 
#cacao anómalo o no
def hay_infeccion(img):
    mapahsl=getmapahsl(img)
    mapahsl_data=mapahsl.reshape(1,1, 45, 64)
    prediccion=modelo_descarte.predict(mapahsl_data)
    return prediccion

#Esta función elimina los gradientes de color y les da un solo valor
def obt_dg(imgx):
    img=imgx.copy()
    if np.amax(img[:,:,0])==0:
        img[:,:,0]=0
    else:
        img[:,:,0]=img[:,:,0]*255/np.amax(img[:,:,0])
        img[:,:,0]=img[:,:,0].astype(int)
    if np.amax(img[:,:,2])==0:
        img[:,:,2]=0
    else:
        img[:,:,2]=img[:,:,2]*255/np.amax(img[:,:,2])
        img[:,:,2]=img[:,:,2].astype(int)
    return img

#Esta funcion reordena el orden de los canales para guardarlo luego
def ord_can(img):
    img_c=img.copy()
    img_c[:,:,2]=img[:,:,0]
    img_c[:,:,0]=img[:,:,2]
    return img_c

#Esta funcion halla el porcentaje de anomalias en la deteccion
def obt_da(img):
    img_c=img.copy()
    img_c[:,:,0][img[:,:,0]!=0]=255
    img_c[:,:,2][img[:,:,2]!=0]=255
    sum_anom=np.sum(img_c[:,:,2])
    sum_normal=np.sum(img_c[:,:,0])
    sum_total=sum_anom+sum_normal
    if int(sum_total)>0:
        porcentaje=np.round(100*sum_anom/(sum_total),2)
    else:
        porcentaje=0
    return img_c,porcentaje

#esta funcion crea una imagen de la zona detectada    
def obt_zd(deteccion,img):
    img_o=img.copy()
    img_o[deteccion[:,:,0]==0]=0
    return img_o
 
#esta funcion crea una imagen con el diagnostico 
def obt_cd(img,diag):
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    h,w,ch=img.shape
    img_c=cv2.putText(img,diag,(20,h-50), fuente, 1,(0,0,255),2,cv2.LINE_AA)
    return img_c
#esta funcion crea una imagen con el diagnostico y el porcentaje
def obt_cd_p(img,diag,porcentaje):
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    h,w,ch=img.shape
    diag=diag+'( '+str(porcentaje)+' %)'
    img_c=cv2.putText(img,diag,(20,h-50), fuente, 1,(0,0,255),2,cv2.LINE_AA)
    return img_c
#esta funcion crea una imagen de la zona no detectada 
def obt_znd(deteccion,img):
    img_o=img.copy()
    img_o[deteccion[:,:,0]!=0]=0
    return img_o
#Esta funcion procesa la imagen para las multiples evaluaciones   
def procesar_imagen(imgx):
#Se reduce el ancho de la imagen a 600 pixeles
    img_origin=imgx
    h,w,ch=img_origin.shape
    f=600/w
    img=cv2.resize(img_origin,None,fx=f,fy=f,interpolation=cv2.INTER_CUBIC)
    h,w,ch=img.shape
#Se crean archivos para guardar los registros
    borrador0=np.zeros((h,w,3))
    borrador0[:,:,1]=1
    borrador1=np.zeros((h,w,4))
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#se da un diagnostico inicial
    diagnostico="No hay Anomalias"
    margen_cacao=0.5
    margen_anomalia=0.5
#se analiza con diferentes ventanas usando el sliding window
    for h_w,w_w in [(90,60),(180,120),(360,240)]:
        for i in range (int(h_w/2),h-int(h_w/2),int(h_w/3)):
            for j in range (int(w_w/2),w-int(w_w/2),int(w_w/3)):
                  rdi=img[int(i-h_w/2):int(i+h_w/2),int(j-w_w/2):int(j+w_w/2)]
                  pred_pres=hay_cacao(rdi)
                  rdi_b=borrador0[int(i-h_w/2):int(i+h_w/2),int(j-w_w/2):int(j+w_w/2)]
				#Se evalua usando las funciones anteriores
                  if pred_pres>margen_cacao:
                     rdi_b[:,:,0]=rdi_b[:,:,0]+1
                     borrador1[i,j,0]=pred_pres
                     borrador1[i,j,2]=h_w
                     borrador1[i,j,3]=w_w
                     pred_anom=hay_infeccion(rdi)
                     if pred_anom>margen_anomalia:
                         rdi_b[:,:,2]=rdi_b[:,:,2]+1
                         rdi_b[:,:,0]=rdi_b[:,:,0]-1
                         borrador1[i,j,1]=pred_anom
                         diagnostico="Hay Anomalias"
    anom_map=borrador0[:,:,2].copy()
    img_r=img.copy()
    img_r[img_r<128]=0
    img_r[img_r>128]=img_r[img_r>128]-128
    img_r[anom_map!=0]=img[anom_map!=0]
    media_d=int(0.25*np.amax(borrador0[:,:,0]))
    media_a=int(0.25*np.amax(borrador0[:,:,2]))
    borrador0[:,:,0][borrador0[:,:,0]<media_d]=0
    borrador0[:,:,2][borrador0[:,:,2]<media_a]=0
    return img_r,borrador0,borrador1,img,diagnostico

#Se cargan los modelos entrenados ganadores
modelo_deteccion=models.load_model("modelo_deteccion.hdf5")
modelo_descarte=models.load_model("modelo_infeccion.hdf5")
counter=0
carpeta_out='data_out/'
carpeta_in='data_in/'
diagnostico="No hay Anomalias"

#Se analiza todas las imágenes de la carpeta de entrada de 
#imágenes y se da el resultado en la carpeta de salida.
for filename in os.listdir(carpeta_in):
    img=cv2.imread(carpeta_in+filename)
    print(counter)
    resultado,deteccion,data,img_o,diagnostico=procesar_imagen(img)
    deteccion_gr=obt_dg(deteccion)
    deteccion_ab,porcentaje=obt_da(deteccion)
    resultado=ord_can(resultado)
    zona_d=ord_can(obt_zd(deteccion,img_o))
    zona_nd=ord_can(obt_znd(deteccion,img_o))
    img_o=ord_can(img_o)
    img_r_cd=obt_cd(resultado.copy(),diagnostico)
    img_r_cdp=obt_cd_p(resultado.copy(),diagnostico,porcentaje)
    cv2.imwrite(carpeta_out+str(counter)+'_dg.png',deteccion_gr)
    cv2.imwrite(carpeta_out+str(counter)+'_da.png',deteccion_ab)
    cv2.imwrite(carpeta_out+str(counter)+'_r.png',resultado)
    cv2.imwrite(carpeta_out+str(counter)+'_zd.png',zona_d)
    cv2.imwrite(carpeta_out+str(counter)+'_znd.png',zona_nd)
    cv2.imwrite(carpeta_out+str(counter)+'_o.png',img_o)
    cv2.imwrite(carpeta_out+str(counter)+'_cdp.png',img_r_cdp)
    cv2.imwrite(carpeta_out+str(counter)+'_rd.png',img_r_cd)
    counter=counter+1

