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

    malware_dir = Args.maldir
    goodware_dir = Args.gooddir
    ncpu_cores = Args.ncpucores

    Logger.debug("malware_dir: {}, goodware_dir: {}, ncpu-cores: {}"
                 .format(malware_dir, goodware_dir, ncpu_cores))
    GetApkData(ncpu_cores, malware_dir, goodware_dir)


def ParseArgs():
    Args = argparse.ArgumentParser(description="Classification of Android Applications")
    Args.add_argument("--maldir", default="../data/apks/malware",
                      help="Absolute path to directory containing malware apks")
    Args.add_argument("--gooddir", default="../data/apks/goodware",
                      help="Absolute path to directory containing benign apks")
    Args.add_argument("--ncpucores", type=int, default=psutil.cpu_count(),
                      help="Number of CPUs that will be used for processing")

    return Args.parse_args()


if __name__ == "__main__":
    main(ParseArgs())
