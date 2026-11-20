"""Carga el archivo local o lo descarga desde una Release pública de GitHub."""
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import hashlib
import json
import os
import tempfile


def verificar(path, esperado):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloque)
    if h.hexdigest() != esperado:
        raise ValueError('El archivo del modelo no coincide con la exportación. Revisa la Release y modelo_sha256.json.')


def preparar_archivo_modelo(carpeta, url=''):
    carpeta = Path(carpeta)
    destino = carpeta / 'modelo_ampliado.tabpfn_fit'
    manifiesto = carpeta / 'modelo_sha256.json'
    esperado = json.loads(manifiesto.read_text())['sha256'] if manifiesto.exists() else None
    if destino.exists():
        if esperado:
            verificar(destino, esperado)
        return destino
    if not url:
        raise FileNotFoundError('Falta modelo_ampliado.tabpfn_fit. Añádelo o configura MODEL_URL con el enlace de su Release en GitHub.')
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'github.com' or '/releases/download/' not in parsed.path:
        raise ValueError('MODEL_URL debe ser el enlace HTTPS directo al archivo de una Release pública de GitHub.')
    if not esperado:
        raise FileNotFoundError('Falta modelo_sha256.json, generado por preparar_github_colab.py.')
    fd, nombre = tempfile.mkstemp(prefix='descarga_modelo_', dir=carpeta)
    try:
        with os.fdopen(fd, 'wb') as salida:
            req = Request(url, headers={'User-Agent': 'Steam-Alcance-App'})
            with urlopen(req, timeout=60) as entrada:
                total = 0
                while True:
                    bloque = entrada.read(1024 * 1024)
                    if not bloque:
                        break
                    total += len(bloque)
                    if total > 2 * 1024**3:
                        raise ValueError('El archivo supera 2 GiB; revisa la exportación.')
                    salida.write(bloque)
        verificar(nombre, esperado)
        os.replace(nombre, destino)
    finally:
        Path(nombre).unlink(missing_ok=True)
    return destino
