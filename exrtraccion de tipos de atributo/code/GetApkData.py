# coding=utf-8
import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

import os
from os.path import join as join_dir
import shutil
import time
import re
import multiprocessing as mp
import CommonModules as CM
from CommonModules import logger
from xml.dom import minidom  # mini Document Object Model for XML
import collections

sys.path.insert(0, "/home/lechu/Documents/UL/2020-1/PoC/frameworks/clasificacion/drebin_mldroid/src/Modules")
import PScoutMapping as PScoutMapping
import BasicBlockAttrBuilder as BasicBlockAttrBuilder

sys.path.insert(0, "/home/lechu/Documents/UL/2020-1/PoC/frameworks/clasificacion/drebin_mldroid/src/Androguard")
import androlyze


def generar_archivo_txt(DataDictionary, filepath, ApkDirectoryPath):
    # Separadores para dar formato a archivo .txt de características
    global featurefile
    usedfeatureSep = "ListaUsesFeatures"
    requestedpermissionSep = "ListaPermisosSolicitados"
    activitySep = "ListaComponenteActividades"
    broadcastreceiverSep = "ListaComponenteReceptorMensajes"
    contentproviderSep = "ListaComponenteProveedorContenido"
    serviceSep = "ListaComponenteServicios"
    filteredintentSep = "ListaFiltrosIntent"
    suspectedapicallSep = "ListaLlamadasApiSospechosas"
    usedpermissionSep = "ListaPermisosUsados"
    restrictedapicallSep = "ListaLlamadasApiRestringidas"
    urlSep = "ListaDominiosUrl"

    # ruta de la carpeta vector_txt
    vector_txt = join_dir(join_dir(ApkDirectoryPath, "vector_txt/"))

    try:
        # crea carpeta vector_formato2
        if not os.path.exists(vector_txt):
            os.makedirs(vector_txt)
        # crea archivo de caracteristicas con nuevo formato
        featurefile = open(filepath, "wb")
        for key, value in DataDictionary.items():
            for v in value:
                if key == usedfeatureSep:
                    print >> featurefile, "uses_feature::" + str(v)
                elif key == requestedpermissionSep:
                    print >> featurefile, "permisos_solicitado::" + str(v)
                elif key == activitySep:
                    print >> featurefile, "componente_actividades::" + str(v)
                elif key == broadcastreceiverSep:
                    print >> featurefile, "componente_receptor_mensajes::" + str(v)
                elif key == contentproviderSep:
                    print >> featurefile, "componente_proveedor_contenido::" + str(v)
                elif key == serviceSep:
                    print >> featurefile, "componente_servicios::" + str(v)
                elif key == filteredintentSep:
                    print >> featurefile, "filtros_intent::" + str(v)
                elif key == suspectedapicallSep:
                    print >> featurefile, "llamadas_api_sospechosas::" + str(v)
                elif key == usedpermissionSep:
                    print >> featurefile, "permisos_usados::" + str(v)
                elif key == restrictedapicallSep:
                    print >> featurefile, "llamadas_api_restringidas::" + str(v)
                elif key == urlSep:
                    print >> featurefile, "dominios_url::" + str(v)

        # nombre del archivo de caracteristicas con nuevo formato
        vector_txt_name = os.path.split(filepath)[1]
        # cerrar archivo generador  
        featurefile.close()
        # mover archivo .data de la carpeta de malware/goodware a la carpeta vector_txt
        shutil.move(filepath, join_dir(vector_txt, vector_txt_name))

    except Exception as e:
        print("Text data writing Failed.")
        logger.error(e)
        logger.error("Text data writing Failed.")
        if "featurefile" in dir():
            featurefile.close()
    else:
        logger.info("Text data of " + filepath + " written successfully.")
        featurefile.close()


def GetFromXML(ApkDirectoryPath, ApkFile):
    """
    Get requested permission etc. for an ApkFile from Manifest files.
    :param String ApkDirectoryPath
    :param String ApkFile
    :return RequestedPermissionSet
    :rtype Set([String])
    :return ActivitySet
    :rtype Set([String])
    :return ServiceSet
    :rtype Set([String])
    :return ContentProviderSet
    :rtype Set([String])
    :return BroadcastReceiverSet
    :rtype Set([String])
    :return HardwareComponentsSet
    :rtype Set([String])
    :return IntentFilterSet
    :rtype Set([String])
    """
    ApkDirectoryPath = os.path.abspath(ApkDirectoryPath)
    ApkFileName = os.path.splitext(ApkFile)[0]
    # ruta de la carpeta XML
    xmlfiledirectory = join_dir(join_dir(ApkDirectoryPath, "XML/"))
    RequestedPermissionSet = set()
    ActivitySet = set()
    ServiceSet = set()
    ContentProviderSet = set()
    BroadcastReceiverSet = set()
    HardwareComponentsSet = set()
    IntentFilterSet = set()

    # crea carpeta XML en caso no exista
    if not os.path.exists(xmlfiledirectory):
        os.makedirs(xmlfiledirectory)

    try:

        ApkFile = os.path.abspath(ApkFile)
        a = androlyze.APK(ApkFile)
        # ruta donde se creará el archivo .xml del apk
        xmlfilepath = os.path.splitext(ApkFile)[0] + ".xml"
        # nombre del archivo .xml
        xmlfilename = os.path.split(xmlfilepath)[1]
        f = open(xmlfilepath, "w")
        # f = open(os.path.splitext(ApkFile)[0] + ".xml", "w")
        f.write((a.xml["AndroidManifest.xml"].toprettyxml(newl="\n\n")).encode("utf-8"))
        f.close()
        # mover archivo .xml de la carpeta malware/goodware a la carpeta XML
        shutil.move(xmlfilepath, join_dir(xmlfiledirectory, xmlfilename))

    except Exception as e:
        print(e)
        logger.error(e)
        logger.error("Executing Androlyze on " + ApkFile + " to get AndroidManifest.xml Failed.")
        return
    try:
        # f = open(ApkFileName + ".xml", "r")
        f = open(join_dir(xmlfiledirectory, xmlfilename), "r")
        Dom = minidom.parse(f)
        DomCollection = Dom.documentElement

        DomPermission = DomCollection.getElementsByTagName("uses-permission")
        for Permission in DomPermission:
            if Permission.hasAttribute("android:name"):
                RequestedPermissionSet.add(Permission.getAttribute("android:name"))

        DomActivity = DomCollection.getElementsByTagName("activity")
        for Activity in DomActivity:
            if Activity.hasAttribute("android:name"):
                ActivitySet.add(Activity.getAttribute("android:name"))

        DomService = DomCollection.getElementsByTagName("service")
        for Service in DomService:
            if Service.hasAttribute("android:name"):
                ServiceSet.add(Service.getAttribute("android:name"))

        DomContentProvider = DomCollection.getElementsByTagName("provider")
        for Provider in DomContentProvider:
            if Provider.hasAttribute("android:name"):
                ContentProviderSet.add(Provider.getAttribute("android:name"))

        DomBroadcastReceiver = DomCollection.getElementsByTagName("receiver")
        for Receiver in DomBroadcastReceiver:
            if Receiver.hasAttribute("android:name"):
                BroadcastReceiverSet.add(Receiver.getAttribute("android:name"))

        DomHardwareComponent = DomCollection.getElementsByTagName("uses-feature")
        for HardwareComponent in DomHardwareComponent:
            if HardwareComponent.hasAttribute("android:name"):
                HardwareComponentsSet.add(HardwareComponent.getAttribute("android:name"))

        DomIntentFilter = DomCollection.getElementsByTagName("intent-filter")
        DomIntentFilterAction = DomCollection.getElementsByTagName("action")
        for Action in DomIntentFilterAction:
            if Action.hasAttribute("android:name"):
                IntentFilterSet.add(Action.getAttribute("android:name"))


    except Exception as e:
        logger.error(e)
        logger.error("Cannot resolve " + DestinationFolder + "'s AndroidManifest.xml File!");
        return RequestedPermissionSet, ActivitySet, ServiceSet, ContentProviderSet, BroadcastReceiverSet, HardwareComponentsSet, IntentFilterSet
    finally:
        f.close()
        return RequestedPermissionSet, ActivitySet, ServiceSet, ContentProviderSet, BroadcastReceiverSet, HardwareComponentsSet, IntentFilterSet


def GetFromInstructions(ApkDirectoryPath, ApkFile, PMap, RequestedPermissionList):
    """
    Get required permissions, used Apis and HTTP information for an ApkFile.
    Reloaded version of GetPermissions.

    :param String ApkDirectoryPath
    :param String ApkFile
    :param PScoutMapping.PScoutMapping PMap
    :param RequestedPermissionList List([String])
    :return UsedPermissions
    :rtype Set([String])
    :return RestrictedApiSet
    :rtype Set([String])
    :return SuspiciousApiSet
    :rtype Set([String])
    :return URLDomainSet
    :rtype Set([String])
    """

    UsedPermissions = set()
    RestrictedApiSet = set()
    SuspiciousApiSet = set()
    URLDomainSet = set()
    try:
        ApkFile = os.path.abspath(ApkFile)
        a, d, dx = androlyze.AnalyzeAPK(ApkFile)
    except Exception as e:
        print(e)
        logger.error(e)
        logger.error("Executing Androlyze on " + ApkFile + " Failed.")
        return
    for method in d.get_methods():
        g = dx.get_method(method)
        for BasicBlock in g.get_basic_blocks().get():
            Instructions = BasicBlockAttrBuilder.GetBasicBlockDalvikCode(BasicBlock)
            Apis, SuspiciousApis = BasicBlockAttrBuilder.GetInvokedAndroidApis(Instructions)
            Permissions, RestrictedApis = BasicBlockAttrBuilder.GetPermissionsAndApis(Apis, PMap,
                                                                                      RequestedPermissionList)
            UsedPermissions = UsedPermissions.union(Permissions)
            RestrictedApiSet = RestrictedApiSet.union(RestrictedApis)
            SuspiciousApiSet = SuspiciousApiSet.union(SuspiciousApis)
            for Instruction in Instructions:
                URLSearch = re.search("https?://([\da-z\.-]+\.[a-z\.]{2, 6}|[\d.]+)[^'\"]*", Instruction, re.IGNORECASE)
                if (URLSearch):
                    URL = URLSearch.group()
                    Domain = re.sub("https?://(.*)", "\g<1>",
                                    re.search("https?://([^/:\\\\]*)", URL, re.IGNORECASE).group(), 0, re.IGNORECASE)
                    URLDomainSet.add(Domain)
    # Got Set S6, S5, S7 described in Drebian paper
    return UsedPermissions, RestrictedApiSet, SuspiciousApiSet, URLDomainSet


def ProcessingDataForGetApkData(ApkDirectoryPath, ApkFile, PMap):
    """
    Produce .data file for a given ApkFile.

    :param String ApkDirectoryPath: absolute path of the ApkFile directory
    :param String ApkFile: absolute path of the ApkFile
    :param PScoutMapping.PScoutMapping() PMap: PMap for API mapping

    :return Tuple(String, Boolean)  ProcessingResult: The processing result, (ApkFile, True/False)
    True means successful. False means unsuccessful.
    """
    vector_data = join_dir(join_dir(ApkDirectoryPath, "vector_data/"))
    try:
        StartTime = time.time()
        logger.info("Start to process " + ApkFile + "...")
        print("Start to process " + ApkFile + "...")
        DataDictionary = {}
        RequestedPermissionSet, ActivitySet, ServiceSet, ContentProviderSet, BroadcastReceiverSet, HardwareComponentsSet, \
        IntentFilterSet = GetFromXML(ApkDirectoryPath, ApkFile)
        RequestedPermissionList = list(RequestedPermissionSet)
        ActivityList = list(ActivitySet)
        ServiceList = list(ServiceSet)
        ContentProviderList = list(ContentProviderSet)
        BroadcastReceiverList = list(BroadcastReceiverSet)
        HardwareComponentsList = list(HardwareComponentsSet)
        IntentFilterList = list(IntentFilterSet)
        DataDictionary["ListaPermisosSolicitados"] = RequestedPermissionList
        DataDictionary["ListaComponenteActividades"] = ActivityList
        DataDictionary["ListaComponenteServicios"] = ServiceList
        DataDictionary["ListaComponenteProveedorContenido"] = ContentProviderList
        DataDictionary["ListaComponenteReceptorMensajes"] = BroadcastReceiverList
        DataDictionary["ListaUsesFeatures"] = HardwareComponentsList
        DataDictionary["ListaFiltrosIntent"] = IntentFilterList
        # Got Set S2 and others

        UsedPermissions, RestrictedApiSet, SuspiciousApiSet, URLDomainSet = GetFromInstructions(ApkDirectoryPath,
                                                                                                ApkFile, PMap,
                                                                                                RequestedPermissionList)
        UsedPermissionsList = list(UsedPermissions)
        RestrictedApiList = list(RestrictedApiSet)
        SuspiciousApiList = list(SuspiciousApiSet)
        URLDomainList = list(URLDomainSet)
        DataDictionary["ListaPermisosUsados"] = UsedPermissionsList
        DataDictionary["ListaLlamadasApiRestringidas"] = RestrictedApiList
        DataDictionary["ListaLlamadasApiSospechosas"] = SuspiciousApiList
        DataDictionary["ListaDominiosUrl"] = URLDomainList

        # ordenar diccionario
        od = collections.OrderedDict(sorted(DataDictionary.items()))

        # crea carpeta vector_data en caso no exista
        if not os.path.exists(vector_data):
            os.makedirs(vector_data)

        # ruta donde se creará el archivo .data del apk
        vector_txt_filepath = os.path.splitext(ApkFile)[0] + ".data"
        # CM.ExportToJson(os.path.splitext(ApkFile)[0] + ".data", DataDictionary)
        CM.ExportToJson(vector_txt_filepath, od)
        # nombre del archivo .data
        vector_data_name = os.path.split(vector_txt_filepath)[1]
        # mover archivo .data de la carpeta malware/goodware a la carpeta vector_data
        shutil.move(vector_txt_filepath, join_dir(vector_data, vector_data_name))
        # ruta del archivo sin extensión .data
        filepath = os.path.splitext(ApkFile)[0]
        # llamar función para generar archivo .txt con nuevo formato
        generar_archivo_txt(od, filepath, ApkDirectoryPath)

    except Exception as e:
        FinalTime = time.time()
        logger.error(e)
        logger.error(ApkFile + " processing failed in " + str(FinalTime - StartTime) + "s...")
        print(ApkFile + " processing failed in " + str(FinalTime - StartTime) + "s...")
        return ApkFile, False
    else:
        FinalTime = time.time()
        logger.info(ApkFile + " processed successfully in " + str(FinalTime - StartTime) + "s")
        print(ApkFile + " processed successfully in " + str(FinalTime - StartTime) + "s")
        return ApkFile, True


def GetApkData(ProcessNumber, *ApkDirectoryPaths):
    """
    Get Apk data dictionary for all Apk files under ApkDirectoryPath and store them in ApkDirectoryPath
    Used for next step's classification

    :param Tuple<string> *ApkDirectoryPaths: absolute path of the directories contained Apk files
    """
    ApkFileList = []
    for ApkDirectoryPath in ApkDirectoryPaths:
        ApkFileList.extend(CM.ListApkFiles(ApkDirectoryPath))
        ApkFileList.extend(CM.ListFiles(ApkDirectoryPath, ""))
    # Because some apk files may not have extension....
    CWD = os.getcwd()
    os.chdir(os.path.join(CWD, "Modules"))
    ''' Change current working directory to import the mapping '''
    PMap = PScoutMapping.PScoutMapping()
    os.chdir(CWD)
    pool = mp.Pool(int(ProcessNumber))
    ProcessingResults = []
    ScheduledTasks = []
    ProgressBar = CM.ProgressBar()
    for ApkFile in ApkFileList:
        if CM.FileExist(os.path.splitext(ApkFile)[0] + ".data"):
            pass
        else:
            # ProcessingDataForGetApkData(ApkDirectoryPath, ApkFile, PMap)
            ApkDirectoryPath = os.path.split(ApkFile)[0]
            apkname = os.path.split(ApkFile)[1]
            ScheduledTasks.append(ApkFile)
            ProcessingResults = pool.apply_async(ProcessingDataForGetApkData,
                                                 args=(ApkDirectoryPath, ApkFile, PMap),
                                                 callback=ProgressBar.CallbackForProgressBar)
    pool.close()
    if (ProcessingResults):
        ProgressBar.DisplayProgressBar(ProcessingResults, len(ScheduledTasks), type="hour")
    pool.join()

    return
