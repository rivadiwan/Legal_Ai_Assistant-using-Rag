import sys
import importlib
print('python_executable:', sys.executable)
try:
    faiss = importlib.import_module('faiss')
    print('faiss imported, version:', getattr(faiss, '__version__', 'no-version'))
except Exception as e:
    print('faiss import failed:', repr(e))

# Try to find distribution info
try:
    import pkg_resources
    d = None
    for dist in pkg_resources.working_set:
        if dist.key.lower().startswith('faiss'):
            d = dist
            break
    if d:
        print('distribution:', d.key, d.version, d.location)
    else:
        print('faiss distribution not found in working_set')
except Exception as e:
    print('pkg_resources error:', repr(e))
