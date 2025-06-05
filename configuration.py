import json
import pandas as pd
from os.path import join as path


class Configuration:
    def __init__(self):
        self._random_state = 123
        
        ### Folder structure ###
        self._path_home = None # CHANGE ME to your root folder (FL-QUIC-TC)path.
        self._path_datasets = path(self.path_home, "datasets")
        self._path_results = path(self.path_home, "results")
        
        # WEEKS and DAYS in CESNET_QUIC22
        
        W44, W45, W46, W47 = "W-2022-44", "W-2022-45", "W-2022-46", "W-2022-47"
        MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = "1_Mon", "2_Tue", "3_Wed", "4_Thu", "5_Fri", "6_Sat", "7_Sun"
        
        self._weeks = [W44, W45, W46, W47]
        self._days = [MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY]
        
        ### DATASET ###
        # Target class
        self._app = "APP"
        self._category = "CATEGORY"
        self._org = "ORG_ID"
        
        # classes
        self._classes = [
            'instagram',
            'spotify',
            'youtube',
            'google-www',
            'facebook-graph',
            'snapchat',
            'discord'
        ]
        
        ## FEATURES
        
        # FLOWSTATS
        self._flowstats = [
            'DURATION',
            'BYTES',
            'BYTES_REV',
            'PACKETS',
            'PACKETS_REV',
            'PPI_LEN',
            'PPI_DURATION',
            'PPI_ROUNDTRIPS',
            'PHIST_SRC_SIZES_1',
            'PHIST_SRC_SIZES_2',
            'PHIST_SRC_SIZES_3',
            'PHIST_SRC_SIZES_4',
            'PHIST_SRC_SIZES_5',
            'PHIST_SRC_SIZES_6',
            'PHIST_SRC_SIZES_7',
            'PHIST_SRC_SIZES_8',
            'PHIST_DST_SIZES_1',
            'PHIST_DST_SIZES_2',
            'PHIST_DST_SIZES_3',
            'PHIST_DST_SIZES_4',
            'PHIST_DST_SIZES_5',
            'PHIST_DST_SIZES_6',
            'PHIST_DST_SIZES_7',
            'PHIST_DST_SIZES_8',
            'PHIST_SRC_IPT_1',
            'PHIST_SRC_IPT_2',
            'PHIST_SRC_IPT_3',
            'PHIST_SRC_IPT_4',
            'PHIST_SRC_IPT_5',
            'PHIST_SRC_IPT_6',
            'PHIST_SRC_IPT_7',
            'PHIST_SRC_IPT_8',
            'PHIST_DST_IPT_1',
            'PHIST_DST_IPT_2',
            'PHIST_DST_IPT_3',
            'PHIST_DST_IPT_4',
            'PHIST_DST_IPT_5',
            'PHIST_DST_IPT_6',
            'PHIST_DST_IPT_7',
            'PHIST_DST_IPT_8'
            ]

        # PSTATS
        self._pstats = [ 
        'PIAT_1',
        'PIAT_2',
        'PIAT_3',
        'PIAT_4',
        'PIAT_5',
        'PIAT_6',
        'PIAT_7',
        'PIAT_8',
        'PIAT_9',
        'PIAT_10',
        'PIAT_11',
        'PIAT_12',
        'PIAT_13',
        'PIAT_14',
        'PIAT_15',
        'PIAT_16',
        'PIAT_17',
        'PIAT_18',
        'PIAT_19',
        'PIAT_20',
        'PIAT_21',
        'PIAT_22',
        'PIAT_23',
        'PIAT_24',
        'PIAT_25',
        'PIAT_26',
        'PIAT_27',
        'PIAT_28',
        'PIAT_29',
        'PIAT_30',
        'DIR_1',
        'DIR_2',
        'DIR_3',
        'DIR_4',
        'DIR_5',
        'DIR_6',
        'DIR_7',
        'DIR_8',
        'DIR_9',
        'DIR_10',
        'DIR_11',
        'DIR_12',
        'DIR_13',
        'DIR_14',
        'DIR_15',
        'DIR_16',
        'DIR_17',
        'DIR_18',
        'DIR_19',
        'DIR_20',
        'DIR_21',
        'DIR_22',
        'DIR_23',
        'DIR_24',
        'DIR_25',
        'DIR_26',
        'DIR_27',
        'DIR_28',
        'DIR_29',
        'DIR_30',
        'PS_1',
        'PS_2',
        'PS_3',
        'PS_4',
        'PS_5',
        'PS_6',
        'PS_7',
        'PS_8',
        'PS_9',
        'PS_10',
        'PS_11',
        'PS_12',
        'PS_13',
        'PS_14',
        'PS_15',
        'PS_16',
        'PS_17',
        'PS_18',
        'PS_19',
        'PS_20',
        'PS_21',
        'PS_22',
        'PS_23',
        'PS_24',
        'PS_25',
        'PS_26',
        'PS_27',
        'PS_28',
        'PS_29',
        'PS_30']

        # SUBFLOWSTATS
        self._pflowstats = [
            #Core
            'bidirectional_duration_ms', 
            'bidirectional_packets', 
            'bidirectional_bytes', 
            'src2dst_duration_ms', 
            'src2dst_packets', 
            'src2dst_bytes', 
            'dst2src_duration_ms', 
            'dst2src_packets', 
            'dst2src_bytes', 


            #PS
            'bidirectional_min_ps', 
            'bidirectional_mean_ps', 
            'bidirectional_stddev_ps', 
            'bidirectional_max_ps', 
            'src2dst_min_ps', 
            'src2dst_mean_ps', 
            'src2dst_stddev_ps', 
            'src2dst_max_ps', 
            'dst2src_min_ps', 
            'dst2src_mean_ps', 
            'dst2src_stddev_ps', 
            'dst2src_max_ps', 
            
            #PIAT
            'bidirectional_min_piat_ms', 
            'bidirectional_mean_piat_ms', 
            'bidirectional_stddev_piat_ms', 
            'bidirectional_max_piat_ms', 
            'src2dst_min_piat_ms', 
            'src2dst_mean_piat_ms', 
            'src2dst_stddev_piat_ms', 
            'src2dst_max_piat_ms', 
            'dst2src_min_piat_ms', 
            'dst2src_mean_piat_ms', 
            'dst2src_stddev_piat_ms', 
            'dst2src_max_piat_ms'
        ]
        
        # SUBPSTATS
        self._pstats_subdirs = [
            'SRC_PS_1',
            'SRC_PS_2',
            'SRC_PS_3',
            'SRC_PS_4',
            'SRC_PS_5',
            'SRC_PS_6',
            'SRC_PS_7',
            'SRC_PS_8',
            'SRC_PS_9',
            'SRC_PS_10',
            'SRC_PS_11',
            'SRC_PS_12',
            'SRC_PS_13',
            'SRC_PS_14',
            'SRC_PS_15',
            'SRC_PS_16',
            'SRC_PS_17',
            'SRC_PS_18',
            'SRC_PS_19',
            'SRC_PS_20',
            'SRC_PS_21',
            'SRC_PS_22',
            'SRC_PS_23',
            'SRC_PS_24',
            'SRC_PS_25',
            'SRC_PS_26',
            'SRC_PS_27',
            'SRC_PS_28',
            'SRC_PS_29',
            'SRC_PS_30',
            'DST_PS_1',
            'DST_PS_2',
            'DST_PS_3',
            'DST_PS_4',
            'DST_PS_5',
            'DST_PS_6',
            'DST_PS_7',
            'DST_PS_8',
            'DST_PS_9',
            'DST_PS_10',
            'DST_PS_11',
            'DST_PS_12',
            'DST_PS_13',
            'DST_PS_14',
            'DST_PS_15',
            'DST_PS_16',
            'DST_PS_17',
            'DST_PS_18',
            'DST_PS_19',
            'DST_PS_20',
            'DST_PS_21',
            'DST_PS_22',
            'DST_PS_23',
            'DST_PS_24',
            'DST_PS_25',
            'DST_PS_26',
            'DST_PS_27',
            'DST_PS_28',
            'DST_PS_29',
            'DST_PS_30',
            'SRC_PIAT_1',
            'SRC_PIAT_2',
            'SRC_PIAT_3',
            'SRC_PIAT_4',
            'SRC_PIAT_5',
            'SRC_PIAT_6',
            'SRC_PIAT_7',
            'SRC_PIAT_8',
            'SRC_PIAT_9',
            'SRC_PIAT_10',
            'SRC_PIAT_11',
            'SRC_PIAT_12',
            'SRC_PIAT_13',
            'SRC_PIAT_14',
            'SRC_PIAT_15',
            'SRC_PIAT_16',
            'SRC_PIAT_17',
            'SRC_PIAT_18',
            'SRC_PIAT_19',
            'SRC_PIAT_20',
            'SRC_PIAT_21',
            'SRC_PIAT_22',
            'SRC_PIAT_23',
            'SRC_PIAT_24',
            'SRC_PIAT_25',
            'SRC_PIAT_26',
            'SRC_PIAT_27',
            'SRC_PIAT_28',
            'SRC_PIAT_29',
            'SRC_PIAT_30',
            'DST_PIAT_1',
            'DST_PIAT_2',
            'DST_PIAT_3',
            'DST_PIAT_4',
            'DST_PIAT_5',
            'DST_PIAT_6',
            'DST_PIAT_7',
            'DST_PIAT_8',
            'DST_PIAT_9',
            'DST_PIAT_10',
            'DST_PIAT_11',
            'DST_PIAT_12',
            'DST_PIAT_13',
            'DST_PIAT_14',
            'DST_PIAT_15',
            'DST_PIAT_16',
            'DST_PIAT_17',
            'DST_PIAT_18',
            'DST_PIAT_19',
            'DST_PIAT_20',
            'DST_PIAT_21',
            'DST_PIAT_22',
            'DST_PIAT_23',
            'DST_PIAT_24',
            'DST_PIAT_25',
            'DST_PIAT_26',
            'DST_PIAT_27',
            'DST_PIAT_28',
            'DST_PIAT_29',
            'DST_PIAT_30',
        ]
        
        # columns for training (All features)
        self._features = self._flowstats + self._pstats + self._pflowstats + self._pstats_subdirs
        
        ## Remove 0 information features
        self._features.remove("PIAT_1")
        self._features.remove("DIR_1")
        self._features.remove("PHIST_DST_SIZES_1")
        self._features.remove("DST_PS_30")
        self._features.remove("SRC_PIAT_30")
        self._features.remove("DST_PIAT_30")
        
        self._columns = self._features + self.appl
        
        ### CL and FL
        self._client_device = "mps"
        self._clients = 14
        self._server_rounds = 112 # FL rounds
        self._epochs = 10 # Client and CL epochs 
        
        self._batch_size_cl = 1024
        self._batch_size_client = 64
        self._lr_cl = 0.01
        self._lr_client = 0.001
        
        self._fedprox_mu = 0.1
         
    def encode(self, cls):
        return self._class_encoder[cls]
    
    def decode(self, id):
        return self._class_decoder[id]
        
    def load_data(self, dataset, index, columns=None):
        return pd.read_parquet(path(self.path_dataset, dataset, f"day-{index}.parquet"), columns=columns)
    
    def load_mapping(self, name):
        if ".json" not in name:
            name = name + ".json"
        with open(path(self.path_results, "class_labeling", name)) as f:
            mapping = json.load(f)
            if "int_to" in name:
                mapping = {int(k): v for k, v in mapping.items()}
        return mapping
    
    @property
    def rounds(self):
        return self._server_rounds
    
    @property
    def clients(self):
        return self._clients
    
    @property
    def classesid(self):
        cls_to_int = self.load_mapping("cls_to_int")
        return [cls_to_int[x] for x in self._classes]
    
    @property
    def classes(self):
        return self._classes

    @property
    def features(self):
        return self._features

    @property
    def columns(self):
        return self._columns

    @property
    def org(self):
        return self._org

    @property
    def category(self):
        return self._category

    @property
    def app(self):
        return self._app
    
    @property
    def appl(self):
        return [self._app]

    @property
    def pflowstats(self):
        return self._pflowstats

    @property
    def flowstats(self):
        return self._flowstats
    
    @property
    def pstats(self):
        return self._pstats
    
    @property
    def weeks(self):
        return self._weeks
    
    @property
    def days(self):
        return self._days
    
    @property
    def path_home(self):
        return self._path_home
    
    @property
    def path_dataset(self):
        return self._path_datasets
    
    @property
    def path_results(self):
        return self._path_results
    
    @property
    def random_state(self):
        return self._random_state
    
    @property
    def batch_size_client(self):
        return self._batch_size_client
    
    @property
    def batch_size_cl(self):
        return self._batch_size_cl
    
    @property
    def epochs(self):
        return self._epochs
    
    @property
    def lr(self):
        return self._lr_cl
    
    @property
    def client_device(self):
        return self._client_device
    
    
    @property
    def fedprox_mu(self):
        return self._fedprox_mu
    
    