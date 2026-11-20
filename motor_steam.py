import numpy as np
import pandas as pd

TAGS = {
    "platforms": ["windows", "mac", "linux"],
    "genres": ["Action", "Adventure", "Indie", "Casual", "Strategy",
               "RPG", "Simulation", "Sports", "Racing"],
    "categories": ["Single-player", "Multi-player", "Co-op",
                   "Steam Achievements", "Steam Trading Cards"]
}
EXTRAS = [
    "Steam Workshop", "Shared/Split Screen", "Full controller support",
    "Partial Controller Support", "Online Multi-Player", "Local Multi-Player",
    "Online Co-op", "Local Co-op"
]

NUM_ORIG = [
    "price", "log_precio", "anio", "antiguedad", "required_age",
    "achievements", "log_logros", "n_plataformas", "n_generos", "n_categorias"
]
NUM_REV = [
    "log_precio", "antiguedad", "required_age", "log_logros",
    "n_plataformas", "n_generos", "n_categorias"
]
BIN_ORIG = ["english", "gratis"] + [
    f"{campo}:{tag}" for campo, tags in TAGS.items() for tag in tags
]
BIN_REV = BIN_ORIG + [f"categories:{tag}" for tag in EXTRAS]

def columnas(revisado):
    return (NUM_REV, BIN_REV) if revisado else (NUM_ORIG, BIN_ORIG)

def leer_etiquetas(valor):
    if not isinstance(valor, str) or not valor.strip():
        return None
    return {t.strip().casefold() for t in valor.split(";") if t.strip()}

def atributos(x, revisado=False):
    z = pd.DataFrame(index=x.index)
    for col in ["price", "required_age", "achievements", "english"]:
        z[col] = pd.to_numeric(x[col], errors="coerce")

    z["anio"] = pd.to_datetime(x["release_date"], errors="coerce").dt.year
    z["antiguedad"] = (2019 - z["anio"]).clip(lower=0)
    z["log_precio"] = np.log1p(z["price"].clip(lower=0))
    z["log_logros"] = np.log1p(z["achievements"].clip(lower=0))
    # Un precio desconocido no equivale a un juego de pago.
    z["gratis"] = (z["price"] == 0).astype(float).where(z["price"].notna())

    contadores = {
        "platforms": "n_plataformas", "genres": "n_generos",
        "categories": "n_categorias"
    }
    for campo, base in TAGS.items():
        listas = x[campo].map(leer_etiquetas)
        z[contadores[campo]] = listas.map(
            lambda valores: len(valores) if valores is not None else np.nan
        )
        etiquetas = base + (EXTRAS if revisado and campo == "categories" else [])
        for tag in etiquetas:
            z[f"{campo}:{tag}"] = listas.map(
                lambda valores: float(tag.casefold() in valores)
                if valores is not None else np.nan
            )

    nums, bins = columnas(revisado)
    return z[nums + bins]


# Funciones compartidas por el notebook y Streamlit.
# Los atributos originales del catálogo se agregan al generar motor_steam.py.
from pathlib import Path
from threading import RLock
import json
import joblib

ORIGEN = ['price', 'release_date', 'required_age', 'english', 'achievements',
          'platforms', 'genres', 'categories']
CLASES = ['Menos de 20 mil', '20 a <50 mil', '50 a <100 mil',
          '100 a <200 mil', '200 mil o más']
IDS = np.arange(5)
CRITICA = 0

def perfil_juego(precio=9.99, antiguedad=2, ingles=True, edad=0, logros=20,
                 plataformas=('windows',), generos=('Action', 'Indie'),
                 categorias=('Single-player',)):
    """Convierte un estilo propuesto a las mismas ocho entradas del entrenamiento."""
    if not plataformas or not generos or not categorias:
        raise ValueError('Elige al menos una plataforma, un género y una modalidad.')
    if not np.isfinite(precio) or precio < 0 or not 0 <= int(antiguedad) <= 22:
        raise ValueError('Revisa precio y antigüedad de referencia.')
    if edad < 0 or logros < 0:
        raise ValueError('La edad mínima y los logros no pueden ser negativos.')
    return pd.DataFrame([{
        'price': float(precio), 'release_date': f'{2019-int(antiguedad)}-01-01',
        'required_age': int(edad), 'english': int(ingles), 'achievements': int(logros),
        'platforms': ';'.join(plataformas), 'genres': ';'.join(generos),
        'categories': ';'.join(categorias)
    }], columns=ORIGEN)

def probabilidades(modelo, X, lote=1):
    clases = np.asarray(modelo.classes_, dtype=int)
    if set(clases) != set(IDS):
        raise ValueError('El modelo debe conservar las cinco clases.')
    if len(X) == 0:
        return np.empty((0, 5))
    orden = [int(np.flatnonzero(clases == k)[0]) for k in IDS]
    p = np.vstack([np.asarray(modelo.predict_proba(X.iloc[i:i+lote]))[:,orden]
                   for i in range(0,len(X),lote)])
    if not np.isfinite(p).all() or (p < -1e-7).any() or not np.allclose(p.sum(1),1,atol=1e-4):
        raise ValueError('Las probabilidades no son válidas.')
    return p

def tabla_alcance(p):
    return pd.DataFrame({'Rango de propietarios': CLASES,
                         'Probabilidad del modelo (%)': np.asarray(p)*100})

def cargar_modelo(carpeta):
    """Restaura el mismo modelo en CPU, sin reducir el entrenamiento."""
    import os
    os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', 'true')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    from tabpfn.model_loading import load_fitted_tabpfn_model
    from sklearn.pipeline import Pipeline
    import torch
    carpeta = Path(carpeta)
    torch.set_num_threads(2)
    pre = joblib.load(carpeta/'preprocesamiento.joblib')
    pfn = load_fitted_tabpfn_model(carpeta/'modelo_ampliado.tabpfn_fit', device='cpu')
    return Pipeline([('preparar',pre),('modelo',pfn)])

def roi_juego(precio_gbp, supuestos):
    """Escenarios financieros, sin convertir owners en ventas automáticamente."""
    s = supuestos
    for k in ['fraccion_precio_cobrado','fraccion_propietarios_pago','comision_plataforma']:
        if not 0 <= s[k] <= 1:
            raise ValueError(f'{k} debe estar entre 0 y 1.')
    for k in ['gbp_clp','costo_variable_copia_clp','desarrollo_clp','lanzamiento_clp']:
        if not np.isfinite(s[k]) or s[k] < 0:
            raise ValueError(f'{k} debe ser un número no negativo.')
    if not np.isfinite(precio_gbp) or precio_gbp < 0:
        raise ValueError('El precio debe ser no negativo.')
    if s['propietarios_ultima_clase'] < 200000:
        raise ValueError('El escenario de la última clase debe ser al menos 200.000.')
    if precio_gbp == 0:
        return pd.DataFrame(), {'aplicable':False,
            'mensaje':'El juego es gratuito: esta fórmula por venta de copias no estima sus ingresos. Se necesita otro modelo de monetización.'}
    precio_clp = precio_gbp*s['gbp_clp']
    neto_copia = precio_clp*s['fraccion_precio_cobrado']*(1-s['comision_plataforma'])
    margen_copia = neto_copia-s['costo_variable_copia_clp']
    margen_propietario = s['fraccion_propietarios_pago']*margen_copia
    fijo = s['desarrollo_clp']+s['lanzamiento_clp']
    propietarios = np.array([10000,35000,75000,150000,s['propietarios_ultima_clase']],float)
    ventas = propietarios*s['fraccion_propietarios_pago']
    ingresos = ventas*neto_copia
    variables = ventas*s['costo_variable_copia_clp']
    costo = fijo+variables
    utilidad = ingresos-costo
    roi = np.divide(100*utilidad,costo,out=np.full_like(costo,np.nan),where=costo>0)
    t = pd.DataFrame({'Rango':CLASES, 'Propietarios supuestos':propietarios,
        'Copias pagadas supuestas':ventas, 'Ingresos netos CLP':ingresos,
        'Costo fijo CLP':fijo, 'Costo variable CLP':variables,
        'Costo total CLP':costo, 'Resultado CLP':utilidad, 'ROI (%)':roi})
    resumen = {'aplicable':True, 'precio_clp':precio_clp,
        'ingreso_neto_copia':neto_copia, 'margen_copia':margen_copia,
        'margen_propietario':margen_propietario, 'costo_fijo':fijo,
        'propietarios_equilibrio':float(np.ceil(fijo/margen_propietario)) if margen_propietario>0 else np.inf}
    return t,resumen

def capacidad_plan(s):
    if s['horas_por_concepto'] <= 0 or s['horas_mes'] < 0 or s['meses'] <= 0:
        raise ValueError('Revisa horas y meses de capacidad.')
    return int(s['horas_mes']//s['horas_por_concepto'])*int(s['meses'])

def priorizar(meta, p, capacidad):
    if len(meta)!=len(p) or meta['appid'].duplicated().any():
        raise ValueError('Metadatos y probabilidades deben corresponder a juegos únicos.')
    r = meta[['appid','name']].copy()
    r['P(bajo alcance)'] = p[:,CRITICA]
    r['Rango estimado'] = [CLASES[i] for i in p.argmax(1)]
    r = r.sort_values(['P(bajo alcance)','appid'],ascending=[False,True],kind='stable')
    r.insert(0,'Prioridad',np.arange(1,len(r)+1))
    r['Seleccionado'] = r['Prioridad'] <= min(capacidad,len(r))
    r['Acción'] = np.where(r['Seleccionado'],
        'Validar interés del público antes de financiar el siguiente prototipo',
        'Mantener en cartera; sin validación asignada en este ciclo')
    return r

def evaluar_ranking(criticos, k):
    z = np.asarray(criticos,dtype=int)
    if len(z)==0 or not np.isin(z,[0,1]).all():
        raise ValueError('Se requieren etiquetas binarias.')
    k = min(max(int(k),0),len(z)); capturados=int(z[:k].sum()); total=int(z.sum())
    azar=k*float(z.mean())
    return {'Casos':len(z),'Capacidad':k,'Críticos totales':total,
        'Capturados modelo':capturados,'Capturados azar esperados':azar,
        'Adicionales':capturados-azar,'Falsas alarmas':k-capturados,
        'Críticos no seleccionados':total-capturados,
        'Precisión@K':capturados/k if k else np.nan,
        'Captura@K':capturados/total if total else np.nan,
        'Lift':capturados/azar if azar else np.nan}

def roi_priorizacion(capturados, k, s, usar_modelo):
    if not 0 <= capturados <= k or not 0<=s['efectividad']<=1:
        raise ValueError('Capturas o efectividad no válidas.')
    for campo in ['costo_hora_clp','implementacion_clp','mantenimiento_clp','gasto_evitable_clp']:
        if s[campo]<0:raise ValueError('No se aceptan costos negativos.')
    revisiones=k*s['horas_por_concepto']*s['costo_hora_clp']
    tecnico=s['implementacion_clp']+s['mantenimiento_clp'] if usar_modelo else 0
    beneficio=capturados*s['efectividad']*s['gasto_evitable_clp']
    costo=revisiones+tecnico
    return {'Críticos detectados':capturados,'Beneficio anual supuesto CLP':beneficio,
        'Costo de todas las validaciones CLP':revisiones,'Costo técnico CLP':tecnico,
        'Costo anual total CLP':costo,'Beneficio neto CLP':beneficio-costo,
        'ROI anual (%)':100*(beneficio-costo)/costo if costo>0 else np.nan}

def ventana_precio(modelo, perfil, referencia):
    """Explora solo el precio; el estilo y la antigüedad permanecen fijos."""
    columnas=['Precio GBP','P(bajo alcance)','Antecedentes cercanos','Ventana candidata']
    vacia=pd.DataFrame(columns=columnas)
    base=float(perfil.iloc[0]['price'])
    if base<=0:return vacia,'No se explora precio de venta para un juego gratuito.'
    generos=leer_etiquetas(perfil.iloc[0]['genres']) or set()
    principal=next((g.casefold() for g in TAGS['genres'] if g.casefold() in generos),None)
    if principal is None:return vacia,'Faltan antecedentes para ese género en los indicadores usados por el modelo.'
    anio=pd.to_datetime(perfil.iloc[0]['release_date']).year
    mascara=referencia['genres'].map(lambda v:principal in (leer_etiquetas(v) or set()))
    mascara &= pd.to_datetime(referencia['release_date']).dt.year.between(anio-2,anio+2)
    mascara &= referencia['english'].eq(perfil.iloc[0]['english'])
    precios=referencia.loc[mascara & referencia['price'].gt(0),'price']
    if len(precios)<20:return vacia,'No hay 20 antecedentes comparables para proponer un rango.'
    lo=max(.75*base,precios.quantile(.05));hi=min(1.25*base,precios.quantile(.95))
    if hi<=lo:return vacia,'El precio propuesto queda fuera de los rangos respaldados por estos datos.'
    grilla=np.unique(np.round(np.linspace(lo,hi,17),2))
    escenarios=pd.concat([perfil]*len(grilla),ignore_index=True);escenarios['price']=grilla
    p=probabilidades(modelo,escenarios)[:,0]
    p_base=float(probabilidades(modelo,perfil)[0,0])
    n=np.array([precios.between(v-max(.5,.15*v),v+max(.5,.15*v)).sum() for v in grilla])
    t=pd.DataFrame({'Precio GBP':grilla,'P(bajo alcance)':p,'Antecedentes cercanos':n,
                    'Ventana candidata':False})
    validos=t.index[n>=20]
    if len(validos)==0:return t,'Ningún precio tiene suficientes antecedentes cercanos.'
    mejor=t.loc[validos,'P(bajo alcance)'].idxmin();minimo=float(t.loc[mejor,'P(bajo alcance)'])
    if minimo>p_base-.01:return t,'El modelo no respalda una reducción de al menos 1 punto porcentual; no recomendamos cambiar el precio con esta evidencia.'
    mascara=(n>=20)&(p<=minimo+.02)&(p<=p_base-.01)
    posiciones=np.flatnonzero(mascara)
    grupos=np.split(posiciones,np.where(np.diff(posiciones)>1)[0]+1)
    grupo=next(g for g in grupos if mejor in g)
    t.loc[grupo,'Ventana candidata']=True
    a,b=t.loc[grupo,'Precio GBP'].min(),t.loc[grupo,'Precio GBP'].max()
    return t,(f'Rango candidato £{a:.2f}–£{b:.2f}: el puntaje de bajo alcance pasa de '
        f'{p_base:.1%} a un mínimo de {minimo:.1%}. Es una hipótesis para validar, no una garantía de ventas ni de rentabilidad.')
