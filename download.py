# Use command: python download.py [dataset name]
# Example: If you want to download MATR dataset, please use command: python download.py MATR
# Where to find the datasets: Datasets may be downloaded in the ./dataset file

import os
import sys
import shutil
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path

def download_file_from_github(dataset_name):

    url_map={
        'CALCE': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/CALCE.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144149Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=062e62883561be8d2efd0c2beacdf1f2d6a1f8ce020e854ea0a0bd18222fee60456dcf86ab853d2acea8277d797766a5acd479162cda94361abb37fa9282e700ab2681851d0b100d92b30a2eda22a43ee84f71b0cba507b37af2a8eb89290331f3361b9a0290f96cce128e215d8ee3173669426d7e732be6f4c38372ca83f29618c5edc6a228b9ec315b944e301c61c0e8cf361729b9361b886701aab686f030a1841c1989d37fe7d4a249f922ed91301f24d79f0e58a45e0a4bacb9d40db161bc5ec8dcdf2758fd495af2c15f5660dc68b77d4a19fe5c344250b2446693f82c1c2c3bb323ddb5daf4bf921968882637312d71527acf1f609b4fca1e3a353bd1',
        'HNEI': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9768320/upload/HNEI.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241106%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241106T114547Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=34ae8843cf7a6d1df9a7eab3a20a4ecac62e77f73850f7241a644867704ed6b338f021db12616f33f567c7abfc6723d8b9a5f00380b349a8ee9d024d83e71d17ccf47e478ab1d9b8c924c8789d5a8f1329ed132dc0ec8ce4613621d29815e753437f80a5adcfe2da545f0a39dbbd749d420f197a170746f7c5e5a15e413c68589ec8e95e4f0af9cbd9eb07d84134c793f341dc83b08e354462176f2f435c324655315c5efb90b1614359382ab030a50d89d6a4d53c1cba3f6c6572db1b084ba3e989b3d4074576860db14df8985d38ce4180fdc4227a91d1d9c5126a76243e310b25d68f26a326a8ed52daed036364e59ab8ace581b20bd81ff14ad3194fb99e',
        'HUST': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/HUST.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144250Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=94e6818daf73cfe19a45179f625b951179c9d787048a4191fbb7bc0484d3619e0cd0b1d8413ac1a70247fc2f189f0e19022c320de0ff7113b1d3c123c4f5d04154d24e63c408bea6119071eb8a23e0b1b5004bb07454c98015d8ffe46697c4049c3f43ba796970b7f278d384d4cb47d187b0bbbafb0f127eab2068f0382c7330335ca3219f477769bf57fe39db14e8f9c7ac23d49aced5ebe017d71054b3fdc0d526b51c8e7cd817bb03eef6a4a40b14525cb77aa5cbc93d829dbe3e119eeb58f5b3201a30390af3a94959bfd6dc8b5a28d280494acec64d602e6fac07bebe61a14a947e3a93f0ad6e91d6ed0c1c624efda931e7428157b0b37692654c7034db',
        'MATR': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/MATR.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144344Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5f682c3f5481d1479efb93d5336351bb32793f3dc5c67cfb7fd23d1f6f689aa773c51c99ab0e06505ecb9fc7ad3282f70fffc3e7a30093b1b50df62a8254522f84449c7fdc957c27e2be6434ad68758d21a3ac2bde6f014a329edf25aa4b091bc829e606a5ae60bce45f91a17fd216fc39aa92cc4383402e3ac340d90e1cdc545bd0697b96f99970d9d4222e7a822ff5a27ce46f703a63a64ddc2a8e029ebfe5d9ebf4962d7eb602caedd2a5828ef2e7fb32b2e3129c4d7fff6bed488e91d505ddba57a1ab5e24f30a7cd22533a4c9d58ed960f153673785c403cd9d30869dce4ee5b887b1aaafa1b032faa33088b86659c05cc6b60e7b6b8f3bc89132e51e69',
        'RWTH': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/RWTH.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144529Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=146249b923c7d4913ebdf1732e3e074860c6900e7871b3f6e9ba0a3ff8566c0a1d3cbe16db61409123c17c8a4d66df20fd8b7366dc617eca124c973fde0558752a57cc543e17cc9a5b7aa77f399b390be155e9014ccbb32c2377f29246122667b88660309c7ec9c3bad3f0f714bd557333d13d3535564baf48d1926610673c6bdcbf9b378c12de5e09a038520edc2fb65a67c04a1f85a3bdb270bc643407a64592195333cf8e4068d9f356b1485c3e3e5af50488e9e902bb1551e2ff82119cd6c296afd7b1028e1009e3dc5b2ffb81394c3c111453ea8be89865589b4f40b2d53fe76ee80bc3d50c8a359d8316381f0c22b4a055ba346b3bc72eaf5c4773ec72',
        'SNL': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/SNL.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144542Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=06fc28590dfd0aa7234b0cbe54c588c1d60673d7a7590d566a6f2e0a0c3d5f05275c3c1fbdcf61893fe37bb8face515a884eb101b13ecf903dc377811ab7045afa3c3abebf6537f55efb1b4263827bfee9b995263a3b0ca67ea234514a7b091c0865cbfbfd79a4966c1d683d1c4dd55e10fdab84396430a2b226308bab240c92093cebe87f2bb50e7d0da52b27782c320f5a9bc5d62fd4258f4846d0cda865f1f7e24938044dd6fe0b6b9dd4456f348ac6cc9ebcb672bb06f679d3eb9ef7b757e8dd61a0b89be7d7475577ce9a4e7c1eca3dbc25d3a94e74e9dfc9b6fb21c273b9e1075874215998f7261d77aa7e45eeb8799917f6b16efb0c1289ef32c67821',
        'MICH': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/MICH.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144405Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5066be86edeb5689dee1fe8a744a9af6e4981172513e4fd73479b26cdae6aec0bd678c946f83c17a8737b910fcb89a05ea9b3fd750305e305adaf9528b1e0242a27a2fded5f4ab423493735a07a270b97ac4924693cdc243dc372fc86ed51f060bd1b5e32d1c01947a75d5759de12a72c5204e4cf798dfc4fd07ab4ce44ebfd07053cfc06d1657b1cca7f5a98a7c2df91c6bc31cdb017315e6cd717a53c25470def1641d035df667c4bf0052b6a270c7b9abb4e758ec8e232eea653c1ba7592e17b190fe4bb4b8d09435281f6a0fd916a857f4a3d1563c952ac503ce4226108f3fd9bd0d048ab4c95858619bba0af66c3b7235f4d9e97dc459bb2899a8f10e49',
        'MICH_EXP': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9829030/upload/MICH_EXP.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241107%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241107T023223Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8e9a06caff94a0ca337b96e7fe71ad5ec44783a62dc38afdb71ce15b2aa0949a838c2a7c7f98ae4ed52b719015450faef245f0ccfc7a64f27a3d3343c531abb5aeee503c491d3531f1e401efadc594e3f2c2159fc8c93b2a077baaef01fc08e9b389180cd7b1fdee74a600f019993e7e3083c87f693e3ad6c08bb7bf9d81edc6bbef76815ca79889acfe7f9f5a52de590724fb5c5a3c83808d40bcac0c83f0789868981fdac8b26dca33ac00a8f0b10e23a9a5068dbf8727c80833db201fb11e79c3b9858d3b28b8f9597512f2e6f39ad55342a4786e0c3ca36f153b9f97b249e3baa7c743ac04202dd562dfb42d9f811428d4d2c25434e2c42e08b954993e70',
        'Tongji': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/Tongji.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144623Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5e9516e9bf02d5ddc70b54746ee29286cabae6ac8188c47823e2946139a9d4501640024a7d235ef233646093e9f345fd2a678bdb67d788961982e55aa3d65ab46af5b4a16403955f104f927da94a3c43c00b80c182dece52bb00f7d0781ba9950a0f9cadaab038fbc2d80a5c81dde3d96a11bae8a243cd4f8de8b21f0f69e021882e8d9cc67745e86f04e1f789def206374168839c9186638b2e53d185bf0f52558c71965dee2b2e9abe7aca45bbe5bd9d03e4cce8c9518e27817e982161feef3a8710c22d1c6735dcc09f634ff765c52e6ed24efb3e61479d5da0e6b56141fd0e1dab378d12d1a5b75f831e8797aabba45581c96b1a7550fbc890a2ba222ff5',
        'Stanford': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/Stanford.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144601Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=659ba98c5e1f2acff7bb0c90accde9228c68f2d39edbb5ffb182e64ea09342e0cd311c29b9a710d17a278d1aab41a0968a71e918c40aa512f83600d03a14a8d991a941cad51a9efdd43c0ac731aecb48ae91116f52413b5fb6b9e4e7ed9e952921f6fc87c9d61cc0baa3f825bab56b0707b9fe1c42491b8095f5aff0c95bb4b4fada10f32107019d0b2ae7860213db606f30a5ee8e78f1e5cdd0bd478b8ff0d3fc4d1e80490faeb6d22933c9f456d841a41a2857e088e0443306322f28be872647b80a4f8f2208572ba398aac25b71fcd50008a9d7f567476a1a89d5d6740e4205d62e7128feebda16239d66e21c8fa44199161b18284fcb453a6802f1d0cd38',
        'ISU_ILCC': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/ISU_ILCC.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144312Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3a0c263daa995a22d09fc4e99e62b86700ca2ed4d0074bcefe2890f60ff0526ee4cfbc64d6edaec975d1fa16911ee5add8d19b67a40a41eaf7b7444211096fe2299a0fb0a12bcd0a8271718f6524119c1590651fa233f5870478a31ff564163e3d4bb484df737b6a92e9b5ccb67553af99bce63e1f41efa54868db86e96f2de0fcfda30ebdcac9fd59e65b892a4ecf5593e91acd244b5888302e58ad097a4621bdc9395f846a1139b0e887c48de595dda4c1ca4515ba2798026a04cbd8c59ea9c17267b9b8ed1d6b3a14c2f1eaf064fbb709abea7734fbcfa5765b1394a46dd345657e5b96c97c7a25c659868997c75a1912ff7e5ddc36add552678e43dc4776',
        'XJTU': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9768320/upload/XJTU.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241031%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241031T030817Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=25a656d592f394385d9a34db84d901586a9b208318dce7eb593c136fa73da373ee44cd9d7e1434f105112eaf0060b57b191a0aeec6b280bde706a59e7eb2fd6cd8df1916fe95889b7119a3f337fe0b66e1d0b9be38684648f3a4f5d7d1be697c36d8ce0b8f2486ea142f4bac1888bfcf2c6f3f711cca5db8e2908aabacff04016678940c211b9c364359d9d7d2d0c58144188316bb3bf97f6a78b3aa280fc7a57d9bc50be6a37bd7203c87aa7c216021e488338284d9d0d22dbbd0895d0b791de8f1b74fda9d579f0b243ad219ac9b2ca0d34e332272b32194c245ef6fd595b3a4fd742bb60ff538015f2708a5155c0d1fc763bf900eb4777b556b282ea491cd',
        'ZN-coin': 'https://storage.googleapis.com/kaggle-data-sets/5677689/10075479/upload/ZNcoin.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241202%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241202T094153Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=86e0f44166aad7dc8ac441c75480b447d24ea7e7d7970d2049e5235a33523a0811c99cff3b8386cf2fe4b05ee6dd9f3cdbe0eb4e901aa1162fb288cd7cdd595e929f0960a80be4d6f17bebe7b6f77aa55f99ca23d2a4a9b5121c9adfca091940ed19587c9dfd5e7722e07d41967acb096b66a07c5c9a254565663496c347eeadd1b3b27ade4785d4b0c6f675de9709cde50bb3f640cd3431cc0b62487e823ac0e74fb886270740b81eafa21b03cf4c8b7dc9d438683f1ef6e6c36a3a03caa16f68d1b0d3adfc15d685c78c29b31fed9edc738ed9e6bf3c5f0a2754ce27ee01fb416a12d581903bc8c73a4367817592c9452eedbe9aba83a8181e490a63e1eece',
        'UL-PUR': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9768320/upload/UL_PUR.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241104%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241104T124625Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=708e78aad159c14654075ad5ae7392c8b2cc1949ea714082fc66331650f7f588d3f5e039fd5abbe95fc4269429abbfa2d61548caeaed10bc2e8733f583f265fd1a27f4374644f9b8410428fdeeb62425f83ecc7bb2e469e2e646120beac4a186c12d5a9790372a4dff10fecdaa3e65855201816950faa9e1c3cc48832ff7c7fbbe383313613ed1b8bdd0cbc8c4d947398f1eff1c2c882b6730efc82a258ea16c00723f764a0cb500b9f66743c009a3e8914b2640f40e1147b3fb07114ede5f4c6157fbdeab5b42a76a5cfa36453f6aca93b4a2f7c6eb2db9062da423f66eada637e64f27528f853184c63d3816af95f912e871202bbc800169283196a829cc9e',
        'NA-coin': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9829030/upload/NAcoin.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241107%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241107T023244Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=30cfeba11687177fd8811ef058cdd529dbe25d2a74bfb1aa79e5f6fd1ba71da1b5449c2ab13e17c752b9a32e20024da460a707b10dfa6525a0ef9d7ef660da5b5e8ac3da3ba8484e4e15b8d2924e7789b9578e75148701e99ec39ba029690dbc4efb4f378ae448fd2f3f68834f97e53b62157c65719abd84a1c5e9d4865e0149af9043201747df56cd3f794b06bb504a1de9bb9d755f4bf2efc5aecbf43f5041b959444a00697a1a43307c29b98b53efa4cbab640ca8216d76889da0f13bab336dc85e7fa135fe9c89c8cb90b90904351514342c2aab90e42620cb8ecac53efb7882630e8abf0e51db159dd67a325b28e7930c700eb75ba18b5aae69d9e5fd36',
        'CALB': 'https://storage.googleapis.com/kaggle-data-sets/5677689/10155988/upload/CALB.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241210%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241210T075624Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8f8bfaecef45f4108e62d18689ad3c0a8c8ab78d9e9a8ad45cf3ebe53b75219ed9f43986ec337faa3e491bbeeedec811f18c222219cead63c64cf0f650ce3df3a9c6e096909b2cec4dc773b40dc6001696d4995a574a4434dae00adf4d7d4ffea8e2fa21e95a9e4483edfa8302da21396c491ad2f50f69c37eda9199acdbd971aa454e39e08e7097a2298466ec1bdab511c41c3369b0260d4d4301ef01af3536e4c879c9b749e2b554811894e313b325d0f4e3570948d1d022647b89346cdaa5c6b2add2c14e715ed2ed35daf3c0b5d2535f98e46a7f9b01225df1817ab3d80d90bcd42cab140fe77be8793803846ca7b81fffc71712d2f6c15f804411090b90',

    }
    url = url_map[dataset_name]
    local_file = f'./dataset/{dataset_name}.zip'

    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024)
        with open(local_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

        # progress_bar.close()
        print(f"Data has been downloaded to {local_file}")

        # Special operation for Tongji dataset
        if dataset_name == 'Tongji':
            path = './dataset'
            raw_file = Path(path) / 'Tongji.zip'
            # Unzip the raw file
            with zipfile.ZipFile(raw_file, 'r') as zip_ref:
                tqdm.write('Unzipping Tongji datasets and rename the files.')
                pbar = zip_ref.namelist()
                for file in tqdm(pbar):
                    zip_ref.extract(file, path)

            files_path = os.listdir(path + '/Tongji/')
            files = [i for i in files_path if i.endswith('.pkl')]
            for file in tqdm(files):
                os.rename(path + '/Tongji/' + file, path + '/Tongji/' + file.split('--')[0] + '-#' + file.split('--')[1])

            os.remove(raw_file)
            print('Delete the old version of Tongji dataset,')

            zip_filename = Path(path) / 'Tongji.zip'
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                tqdm.write('Compress the Tongji dataset')
                for file in tqdm(os.listdir(Path(path) / 'Tongji')):
                    file_path = Path(path) / 'Tongji' / file
                    if file_path.is_file():
                        zipf.write(file_path, arcname=file)

            shutil.rmtree(raw_file.parent / 'Tongji')
            print('Complete rename operations.')

    else:
        print(f"Download failed: {response.status_code}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download.py <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    download_file_from_github(dataset_name)
