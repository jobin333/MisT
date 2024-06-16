dataset_details = {
                    'cholec80':{
                        'video_count':80, 
                        'num_classes':7, 
                        'test_file_count': 11,
                    },
    
                    'm2cai16':{
                        'video_count':[27, 14], 
                        'num_classes':8, 
               
                    },
    
                    'autolaparo':{
                        'video_count':21, 
                        'num_classes':7, 
                        'test_file_count': 3,
                    
                    }
                  }

feature_model_details = { 'Swin3D_B':1024, 'Swin3D_S':768, 'Swin3D_T':768 }


configs = {
    'dataset_details': dataset_details,
    'feature_model_details': feature_model_details,
    'feature_folder' : '/workspace/Features',
    'model_save_folder' : '/workspace/Models',
    'random_train_test' : False,
    
    'flm_seq_length' : 30,
    'flm_lr' : 0.0001,
    'flm_max_epoch' : 50,
    
    'slm_lr' : 0.0001,
    'slm_max_epoch' : 50,
    'slm_stack_length' : 60,
    'slm_roll_count' : 20,
    'slm_number_path' : 6,
    'slm_path_multiplier' : 2,
    'slm_dropout' : 0.0
}

