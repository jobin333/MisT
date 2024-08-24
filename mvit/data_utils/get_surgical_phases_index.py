def print_surg_phases():
  cholec80 = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting',
                          'GallbladderDissection', 'GallbladderPackaging',
                          'CleaningCoagulation', 'GallbladderRetraction']

  m2cai16 = ['TrocarPlacement','Preparation','CalotTriangleDissection'
                            ,'ClippingCutting','GallbladderDissection','GallbladderPackaging'
                            ,'CleaningCoagulation','GallbladderRetraction']

  autolaparo = ['Preparation','Dividing Ligament and Peritoneum',
                'Dividing Uterine Vessels and Ligament','Transecting the Vagina',
                'Specimen Removal','Suturing','Washing']


  print('Cholec80 Phases')
  print(sorted(cholec80))
  print('M2CAI16 Phases')
  print(sorted(m2cai16))
  print('AutoLaparo Phases')
  print(autolaparo)

if __name__ == '__main__':
  print_surg_phases()