from importlib import import_module


def import_fws_module(backend, name):
    mdl_path = 'fws.{}.{}'.format(backend, name)
    return import_module(mdl_path)

