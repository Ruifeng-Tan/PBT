class gate_masker:
    MIX_large_temperature2mask = {
        -5.0: [0,1,2],
        15.0: [0,1,2],
        20.0: [1,2,3],
        23.0: [2,3,4,5],
        25.0: [3,4,5,6,7,8,9,10,11],
        30.0: [4,5,6,7,8,9,10,11,12],
        35.0: [6,7,8,9,10,11,12,13],
        45.0: [12,13,14],
        55.0: [12,13,14]
    }

    MIX_large_format2mask = {
        'prismatic': [0],
        'cylindrical': [1,2,3,4,5,6],
        'polymer': [7,8,9],
        'pouch': [10]
    } 

    MIX_large_cathodes2mask = {
        'LFP': [0,1,2],
        'NCA': [3],
        'NCM': [4,5,6,7,8,9],
        'LCO': [10],
        'NCA_NCM': [3,4,5,6,7,8,9],
        'NCM_NCA': [3,4,5,6,7,8,9],
        'LCO_NCM': [4,5,6,7,8,9,10],
        'NCM_LCO': [4,5,6,7,8,9,10]
    }

    MIX_large_anode2mask = {
        'graphite': [0,1,2,3,4,5,6,7,8,9],
        'graphite/Si': [10]
    }

    # MIX_all
    MIX_all_temperature2mask = {
        -5.0: [0,1,2],
        0.0: [0,1,2],
        15.0: [1,2,3],
        20.0: [2,3,4],
        23.0: [3,4,5,6,7,8],
        25.0: [4,5,6,7,8,9,10,11,12,13,14],
        30.0: [5,6,7,8,9,10,11,12,13,14,15],
        35.0: [9,10,11,12,13,14,15,16],
        45.0: [15,16,17],
        55.0: [15,16,17]
    }

    MIX_all_format2mask = {
        'prismatic': [0],
        'cylindrical': [1,2,3,4,5,6],
        'polymer': [7,8,9],
        'pouch': [10],
        'coin': [11]
    } 

    MIX_all_cathodes2mask = {
        'LFP': [0,1,2],
        'NCA': [3],
        'NCM': [4,5,6,7,8,9],
        'LCO': [10],
        'NCA_NCM': [3,4,5,6,7,8,9],
        'NCM_NCA': [3,4,5,6,7,8,9],
        'LCO_NCM': [4,5,6,7,8,9,10],
        'NCM_LCO': [4,5,6,7,8,9,10],
        'MnO2': [11],
        'Unknown': [12]
    }

    MIX_all_anodes2mask = {
        'graphite': [0,1,2,3,4,5,6,7,8,9],
        'graphite/Si': [10],
        'zinc metal': [11],
        'Unknown': [12]
    }