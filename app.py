from pathlib import Path
import json
import os
from threading import RLock
import numpy as np
import pandas as pd
import streamlit as st
import motor_steam as motor
from archivos_modelo import preparar_archivo_modelo

BASE=Path(__file__).resolve().parent
st.set_page_config(page_title='Steam · Alcance de tu juego',page_icon='🎮',layout='wide')

def secreto(nombre, defecto=''):
    try:
        return st.secrets.get(nombre, os.environ.get(nombre, defecto))
    except FileNotFoundError:
        return os.environ.get(nombre, defecto)

token = secreto('TABPFN_TOKEN')
if token:
    os.environ['TABPFN_TOKEN'] = str(token)
os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', 'true')

@st.cache_resource
def archivo_modelo(url):
    return preparar_archivo_modelo(BASE, url)

@st.cache_resource
def recursos(firma):
    return motor.cargar_modelo(BASE),RLock()

@st.cache_data
def datos():
    cfg=json.loads((BASE/'configuracion.json').read_text(encoding='utf-8'))
    train=pd.read_csv(BASE/'referencia_entrenamiento.csv')
    return cfg,train

@st.cache_data(max_entries=128)
def predecir_perfil(_modelo,_candado,perfil,firma):
    # Cambiar el presupuesto no requiere volver a consultar el mismo perfil.
    with _candado:
        return motor.probabilidades(_modelo,perfil)[0]

try:
    cfg,referencia=datos()
    with st.spinner('Preparando el modelo guardado…'):
        ruta_modelo=archivo_modelo(str(secreto('MODEL_URL')))
    firma=ruta_modelo.stat().st_mtime_ns
    with st.spinner('Cargando el TabPFN ampliado en CPU…'):
        modelo,candado=recursos(firma)
except Exception as error:
    st.error(f'No se pudo cargar el modelo: {error}')
    st.info('Revisa los archivos exportados del Hito 2 y la configuración de Secrets siguiendo el README. La app no necesita Colab conectado una vez exportados los archivos.')
    st.stop()

st.title('¿Qué alcance podría tener un juego de este estilo?')
st.caption('TabPFN ajustado, ampliado · 21.660 juegos de entrenamiento · F1 CV histórico: 0,4424')
st.info('Estimamos un rango de propietarios a partir del catálogo histórico de 2019. Es una referencia para evaluar una idea; no una promesa de ventas futuras.')

with st.sidebar, st.form('perfil_del_juego'):
    st.header('Diseña el perfil de tu juego')
    nombre=st.text_input('Nombre del proyecto','Mi juego')
    generos=st.multiselect('Géneros',cfg['opciones']['genres'],default=['Action','Indie'])
    plataformas=st.multiselect('Plataformas',cfg['opciones']['platforms'],default=['windows'])
    categorias=st.multiselect('Modalidades y categorías',cfg['opciones']['categories'],default=['Single-player'])
    precio=st.number_input('Precio de referencia (GBP)',min_value=0.,max_value=float(cfg['precio_max']),value=9.99,step=.5)
    ingles=st.checkbox('Disponible en inglés',value=True)
    with st.expander('Otras características'):
        antiguedad=st.slider('Antigüedad de referencia (años)',0,15,2,
            help='Compara con la antigüedad acumulada de los juegos del catálogo. No es un pronóstico temporal validado.')
        edad=st.selectbox('Edad mínima requerida',[0,3,7,12,16,18],index=0)
        logros=st.number_input('Cantidad de logros',0,10000,20,step=1)
    st.caption('Los precios del dataset están en GBP. Los escenarios financieros se expresan en CLP con un cambio supuesto editable.')
    enviado=st.form_submit_button('Estimar alcance',type='primary')

try:
    perfil=motor.perfil_juego(precio,antiguedad,ingles,edad,logros,plataformas,generos,categorias)
    with st.spinner('Estimando alcance en CPU…'):
        p=predecir_perfil(modelo,candado,perfil,firma)
except ValueError as error:
    st.warning(str(error));st.stop()
except Exception as error:
    st.error(f'No se pudo calcular la predicción: {error}')
    st.info('Esta versión conserva los 21.660 juegos. Si el servidor agota la memoria, los recursos gratuitos no bastan para esta configuración.')
    st.stop()

clave=perfil.to_json()
if st.session_state.get('perfil_ventana')!=clave:
    st.session_state.pop('ventana',None)

sim,finanzas,plan,resultados=st.tabs(['Simulador','ROI del videojuego','Plan de acción','Resultados del modelo'])
with sim:
    st.subheader(nombre)
    a,b=st.columns(2)
    a.metric('Rango con mayor probabilidad',motor.CLASES[int(np.argmax(p))])
    b.metric('Probabilidad del modelo para ese rango',f'{p.max():.1%}')
    t=motor.tabla_alcance(p)
    st.bar_chart(t.set_index('Rango de propietarios'))
    st.dataframe(t.round(2),hide_index=True,width='stretch')
    st.caption('Las cinco probabilidades suman 100 %. No están calibradas; no calculamos una cantidad exacta de público ni un ROI promedio ponderado con ellas.')
    st.subheader('¿Hay un rango de precios que convenga explorar?')
    st.write('El estilo se mantiene fijo. Buscamos menor puntaje de bajo alcance en precios cercanos y con antecedentes del entrenamiento.')
    if st.button('Explorar rango de precios',type='primary'):
        with st.spinner('Comparando precios…'),candado:
            st.session_state['ventana']=motor.ventana_precio(modelo,perfil,referencia)
        st.session_state['perfil_ventana']=clave
    if 'ventana' in st.session_state:
        ventana,texto=st.session_state['ventana'];st.write(texto)
        if len(ventana):
            graf=ventana[['Precio GBP','P(bajo alcance)']].copy();graf['P(bajo alcance)']*=100
            st.line_chart(graf.set_index('Precio GBP'))
            st.dataframe(ventana.round(4),hide_index=True,width='stretch')
    with st.expander('Cómo se elige la ventana'):
        st.write('Hasta ±25 % del precio, dentro del percentil 5–95 de comparables; mínimo 20 antecedentes cercanos. Se exige una reducción de 1 punto porcentual y se acepta una banda hasta 2 puntos sobre el mínimo. Son reglas de exploración, no significancia estadística ni precio óptimo.')

with finanzas:
    st.subheader('¿El alcance podría cubrir el presupuesto?')
    st.write('Todos los montos de esta sección son supuestos. Los propietarios se convierten a copias pagadas solo mediante la fracción editable indicada abajo.')
    s=dict(cfg['supuestos_juego'])
    with st.expander('Editar presupuesto e ingresos',expanded=True):
        c1,c2,c3=st.columns(3)
        s['desarrollo_clp']=c1.number_input('Desarrollo (CLP)',min_value=0,value=int(s['desarrollo_clp']),step=1000000)
        s['lanzamiento_clp']=c2.number_input('Lanzamiento / marketing (CLP)',min_value=0,value=int(s['lanzamiento_clp']),step=1000000)
        s['gbp_clp']=c3.number_input('CLP por GBP · cambio supuesto',min_value=1.,value=float(s['gbp_clp']),step=50.)
        s['fraccion_precio_cobrado']=c1.slider('Precio efectivamente cobrado (%)',0,100,int(100*s['fraccion_precio_cobrado']))/100
        s['fraccion_propietarios_pago']=c2.slider('Propietarios con copia pagada (%)',0,100,int(100*s['fraccion_propietarios_pago']))/100
        s['comision_plataforma']=c3.slider('Comisión de plataforma supuesta (%)',0,100,int(100*s['comision_plataforma']))/100
        s['costo_variable_copia_clp']=c1.number_input('Costo variable por copia pagada (CLP)',min_value=0,value=int(s['costo_variable_copia_clp']),step=100)
        s['propietarios_ultima_clase']=c2.number_input('Escenario para ≥200 mil propietarios',min_value=200000,value=int(s['propietarios_ultima_clase']),step=50000)
    t,res=motor.roi_juego(precio,s)
    if not res['aplicable']:st.warning(res['mensaje'])
    else:
        st.metric('Propietarios necesarios para cubrir costos',f"{res['propietarios_equilibrio']:,.0f}" if np.isfinite(res['propietarios_equilibrio']) else 'No hay equilibrio con margen no positivo')
        st.write(f"Ingreso neto por copia pagada: **{res['ingreso_neto_copia']:,.0f} CLP**. Margen por propietario bajo los supuestos: **{res['margen_propietario']:,.0f} CLP**.")
        t.insert(1,'Probabilidad del modelo (%)',100*p)
        st.dataframe(t.round(2),hide_index=True,width='stretch')
        st.caption('10.000, 35.000, 75.000 y 150.000 son ejemplos dentro de cada rango; no predicciones puntuales. El valor para la última clase es supuesto porque no tiene límite superior. ROI simple antes de impuestos, sin descuento de flujos ni horizonte anual validado.')
        st.download_button('Descargar escenarios del juego',t.to_csv(index=False).encode('utf-8-sig'),'roi_mi_juego.csv','text/csv')

with plan:
    st.subheader('Priorizar qué ideas validar antes de financiar un prototipo')
    capacidad=motor.capacidad_plan(cfg['supuestos_plan'])
    cartera=cfg['supuestos_plan']['propuestas_anuales']
    st.write(f'Ejercicio retrospectivo: {cartera} juegos reservados representan {cartera} propuestas anuales. Se priorizan {capacidad} validaciones por riesgo de quedar bajo 20 mil propietarios. Se revisa el interés del público antes de comprometer el presupuesto de la siguiente etapa.')
    st.dataframe(pd.read_csv(BASE/'ranking_plan.csv').head(capacidad),hide_index=True,width='stretch')
    st.image(str(BASE/'ganancia.png'),caption='Captura de casos de bajo alcance frente al azar, a igual capacidad.')
    st.subheader('ROI anual del sistema de priorización')
    st.caption('Este ROI corresponde al proceso de selección de ideas; es distinto al ROI de desarrollar un videojuego.')
    st.dataframe(pd.read_csv(BASE/'roi_plan.csv').round(2),hide_index=True,width='stretch')
    st.dataframe(pd.read_csv(BASE/'sensibilidad_plan.csv').round(2),hide_index=True,width='stretch')
    with st.expander('Supuestos del plan'):
        st.json(cfg['supuestos_plan'])
    st.markdown((BASE/'conclusion_plan.md').read_text(encoding='utf-8'))

with resultados:
    st.write('Modelo usado en esta v2: **TabPFN ajustado, ampliado**, sin trayectoria ni clasificación ordinal.')
    st.dataframe(pd.read_csv(BASE/'evaluacion_final.csv').round(4),hide_index=True,width='stretch')
    st.image(str(BASE/'confusion_final.png'),caption='Matriz de confusión en la reserva histórica de 3.015 juegos.')
    st.dataframe(pd.read_csv(BASE/'reporte_clases.csv').round(3),hide_index=True,width='stretch')
    st.caption('La reserva ya se consultó durante el proyecto. La evaluación es retrospectiva; para pronosticar lanzamientos nuevos hace falta otra cohorte y un horizonte común de medición.')
