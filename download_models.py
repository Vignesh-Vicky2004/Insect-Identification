import open_clip
import requests
import os

print('Pre-downloading BioCLIP model...')
try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        cache_dir='/app/models'
    )
    print('✅ BioCLIP model cached successfully')
except Exception as e:
    print(f'⚠️ BioCLIP pre-cache failed: {e}')

print('Pre-downloading classifier...')
try:
    url = 'https://github.com/Vignesh-Vicky2004/Insect-Identification/raw/main/best_bioclip_classifier.pth'
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    with open('/app/best_bioclip_classifier.pth', 'wb') as f:
        f.write(response.content)
    print('✅ Classifier cached successfully')
except Exception as e:
    print(f'⚠️ Classifier pre-cache failed: {e}')
