import argparse
import logging
import psutil

from GetApkData import GetApkData

logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('main.stdout')


def main(Args):
    """
    Main function for malware and goodware classification
    :param args: arguments acquired from command lines(refer to ParseArgs() for list of args)
    """

    malware_dir = Args.mw_dir
    goodware_dir = Args.gw_dir
    ncpu_cores = Args.cpu_cores

    Logger.debug("malware_dir: {}, goodware_dir: {}, ncpu_cores: {}"
                 .format(malware_dir, goodware_dir, ncpu_cores))
    GetApkData(ncpu_cores, malware_dir, goodware_dir)


def ParseArgs():
    Args = argparse.ArgumentParser(description="Classification of Android Applications")
    Args.add_argument("--mw_dir", default="../data/apks/malware",
                      help="Ruta del directorio que contiene las aplicaciones (.apks) malware")
    Args.add_argument("--gw_dir", default="../data/apks/goodware",
                      help="Ruta del directorio que contiene las aplicaciones (.apks) benignas")
    Args.add_argument("--cpu_cores", type=int, default=psutil.cpu_count(),
                      help="Maximo numero de nucleos del CPU que se usaran para el multiprocesamiento de las apks "
                           "(parametro usado en la extraccio de tipos de atributo)")
    return Args.parse_args()


if __name__ == "__main__":
    main(ParseArgs())
