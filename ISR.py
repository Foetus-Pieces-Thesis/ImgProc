#Installation
#pip install ISR 
#pip install 'h5py==2.10.0' --force-reinstall

import numpy as np
from PIL import Image


img = Image.open('data/input/test_images/ISRtest.png')

# Uncomment if using local repo
# import sys
# sys.path.append('..')
from ISR.models import RDN, RRDN


#model = RDN(weights='noise-cancel')
#model = RRDN(weights='gans')
#model = RDN(weights='psnr-small')
model = RDN(weights='psnr-large')

#Create Baseline
# img.size[0]*2 : int here tells us zoom factor
# resample: allows us to choose other mathematicl functions
bicubic_img=img.resize(size=(img.size[0]*2, img.size[1]*2), resample=Image.BICUBIC)

#Save
#bicubic_img = Image.fromarray(bicubic_img)
bicubic_img.save('2_bicubic.png')

#Prediction
sr_img = model.predict(np.array(img))
isr_img=Image.fromarray(sr_img)
isr_img.save('2_psnr_large.png')
