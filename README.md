# Alcance de videojuegos en Steam

Proyecto de Hellen Quiñones y Gustavo Sánchez. Aplicación Streamlit con TabPFN ajustado y ampliado sobre 21.660 juegos, simulador de alcance, escenarios de ROI y plan de priorización.

Esta carpeta ya contiene el modelo ajustado, su preprocesamiento y los resultados reales del ZIP exportado del Hito 2. **No hay que volver a Colab ni reentrenar para preparar el repositorio.**

## 1 Crear el repositorio

1. Descomprime `steam_github_listo.zip` en tu computador.
2. Abre https://github.com/new e inicia sesión.
3. Nombre: `steam-alcance-juegos`. Selecciona **Public** y pulsa **Create repository**. Puedes dejar desmarcada la creación de un README, porque ya viene incluido.
4. Selecciona **uploading an existing file** o **Add file → Upload files**.
5. Arrastra el contenido descomprimido, no el ZIP ni una carpeta contenedora. Comprueba que `app.py`, `motor_steam.py`, `requirements.txt`, `modelo_ampliado.tabpfn_fit` y los CSV estén en la raíz.
6. Pulsa **Commit changes**. Todos los archivos de esta entrega son menores a 25 MiB; no necesitas Releases ni Git LFS.

La carpeta `.streamlit` define el tema visual. Si tu explorador la oculta, la app puede funcionar sin ella. No subas archivos de claves como `secrets.toml`.

## 2 Publicar Streamlit gratis

1. Abre https://share.streamlit.io e inicia sesión con GitHub.
2. Pulsa **Create app** y elige el repositorio `steam-alcance-juegos`.
3. Selecciona rama `main` y archivo principal `app.py`.
4. En **Advanced settings**, selecciona **Python 3.12**.
5. En **Secrets**, pega el token de Prior Labs que utilizabas en Colab con este formato:

```toml
TABPFN_TOKEN = "TU_TOKEN_REAL"
```

6. Guarda y pulsa **Deploy**. No necesitas `MODEL_URL` porque el archivo ajustado ya está incluido.

La primera carga instala las dependencias y puede descargar los pesos base de TabPFN mediante su biblioteca. El archivo `.tabpfn_fit` guarda el estado ajustado; su pequeño tamaño no representa toda la memoria necesaria para ejecutar el modelo. No se usa Hugging Face como alojamiento de la aplicación.

## 3 Comprobar el funcionamiento

1. Abre el enlace de la app y estima el perfil inicial.
2. Cambia el género o precio y pulsa **Estimar alcance**.
3. Revisa el ROI, el plan de acción y los resultados.
4. Desconecta Colab y repite una predicción. Si responde, ya comprobaste que funciona de forma independiente.

Si no carga, entra en **Manage app → Logs** y revisa el primer error. Si se agota la RAM, el modelo completo supera los recursos gratuitos disponibles. Esta carpeta permite intentar la ejecución en CPU; todavía no se ha comprobado el consumo y la latencia del modelo real en Community Cloud. La aplicación puede suspenderse por inactividad y necesitar una nueva carga al abrirla.

## Cambios realizados

- Carga del mismo modelo en CPU en lugar de exigir CUDA y una sesión Colab.
- Autorización explícita de CPU para conjuntos grandes, dos hilos y predicciones en lotes de una fila.
- Caché de modelo y predicciones; formulario para confirmar cambios del perfil.
- Token suministrado por Secrets y comprobación de integridad del archivo del modelo.
- Corrección del texto de importes CLP para evitar que se interprete como una fórmula.

Se conservan sin modificar el archivo del modelo, el preprocesamiento, los resultados y los supuestos financieros del ZIP original. Los F1 son los resultados de la ejecución original, no una nueva evaluación del despliegue.

## Fuentes

- Datos: https://www.kaggle.com/datasets/nikdavis/steam-store-games
- TabPFN: https://github.com/PriorLabs/TabPFN
- Despliegue: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Secrets: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

El dataset representa el catálogo de 2019. Los propietarios son estimaciones y no equivalen a ventas exactas. Los ROI utilizan supuestos editables y no son rentabilidades garantizadas.
