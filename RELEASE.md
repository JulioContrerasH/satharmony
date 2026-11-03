# Release 0.2.0

PyPI pública: 0.1.13. Esta es la primera 0.2.0.

## Construir

```bash
rm -rf dist
python -m build
python -m twine check dist/*
```

Salen `dist/satharmony-0.2.0-py3-none-any.whl` y `dist/satharmony-0.2.0.tar.gz`.

## TestPyPI

```bash
python -m twine upload --repository testpypi dist/satharmony-0.2.0*
python -m pip install -i https://test.pypi.org/simple/ satharmony==0.2.0
satharmony --help
```

Las credenciales no viven en el repo. En deep no hay token de TestPyPI, así que
este paso queda pendiente hasta que Julio lo pase.

## PyPI

Solo después de que TestPyPI y el quickstart funcionen:

```bash
python -m twine upload dist/satharmony-0.2.0*
```

## Tag

```bash
git tag v0.2.0
git push origin fix-p0 v0.2.0
```

El push y el tag necesitan permiso de Julio; el CI se dispara solo.
