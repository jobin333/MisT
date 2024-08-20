
dataset_details = {
                    'cholec80':{
                        'video_count':80, 
                        'num_classes':7, 
                        'test_file_count': 11,
                        'surgical_phases': ['Preparation', 'CalotTriangleDissection', 'ClippingCutting',
                        'GallbladderDissection', 'GallbladderPackaging',
                        'CleaningCoagulation', 'GallbladderRetraction']
                    },
    
                    'm2cai16':{
                        'video_count':[27, 14], 
                        'num_classes':8, 
                        'surgical_phases': ['TrocarPlacement','Preparation','CalotTriangleDissection'
                           ,'ClippingCutting','GallbladderDissection','GallbladderPackaging'
                           ,'CleaningCoagulation','GallbladderRetraction']
               
                    },
    
                    'autolaparo':{
                        'video_count':21, 
                        'num_classes':7, 
                        'test_file_count': 3,
                        'surgical_phases': ['Preparation', 'Dividing Ligament and Peritoneum',
                            'Dividing Uterine Vessels and Ligament',
                            'Transecting the Vagina', 'Specimen Removal',
                            'Suturing', 'Washing']
                    
                    }
                  }

feature_model_details = { 'Swin3D_B':1024, 'Swin3D_S':768, 'Swin3D_T':768 }


slm_rolls = [ 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]



configs = {
    'dataset_details': dataset_details,
    'feature_model_details': feature_model_details,
    'feature_folder' : '/workspace/Features',
    'model_save_folder' : '/workspace/Models-test',
    'random_train_test' : False,
    'overwrite_config' : False,
    'random_seed': 0,
    
    'flm_seq_length' : 15,
    'flm_lr' : 0.00002,
    'flm_max_epoch' : 200,
    'flm_stop_epoch_count': 12,
    'flm_model_name': 'Simple_Linear',
    'flm_model_out_path': '/workspace/ModelOut',
    
    # Specific to MultiLevelLinear Model
    'slm_roll_start_with_one': True,
    'slm_roll_count' : 20,
    'slm_path_multiplier' : 2,
    
    
    # General for SLM - Best Accuracy configs
    'slm_model_name': 'SLM',
    'slm_stop_epoch_count': 12,
    'slm_lr' : .00005,
    'slm_max_epoch' : 100,
    'slm_stack_length' : 32,
    'slm_number_path' : len(slm_rolls),
    
    # Specific to MultiLevelTransformer Model
    'slm_dropout' : 0.0,
    'slm_strides': slm_rolls,
    'slm_nhead': 4,
    'slm_dim_feedforward': 128,
    'slm_num_layers': 4,
    'slm_dmodel': 128,

}


