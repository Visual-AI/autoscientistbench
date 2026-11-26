"""Safe DataLoader startup on JAX and long paper-local temporary paths."""
def configure_multiprocessing():
    import multiprocessing as mp
    from multiprocessing import connection
    import uuid, os, pathlib
    mp.set_start_method('forkserver', force=True)
    source_dir=str(pathlib.Path(__file__).resolve().parent)
    os.environ['PYTHONPATH']=source_dir+os.pathsep+os.environ.get('PYTHONPATH','')
    mp.set_forkserver_preload(['forkserver_preload'])
    original=connection.arbitrary_address
    def address(family):
        if family=='AF_UNIX':
            # Linux abstract sockets avoid the 108-byte filesystem pathname limit.
            return b'\0metric-'+uuid.uuid4().hex.encode('ascii')
        return original(family)
    connection.arbitrary_address=address
