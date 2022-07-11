import glob
import os
image_all = glob.glob('data/VOCdevkit/VOC2012/JPEGImages/*.jpg')
image_all_name = [image_file.replace('\\', '/').split('/')[-1].split('.')[0] for image_file in image_all]

image_SegmentationClass = glob.glob('data/VOCdevkit/VOC2012/SegmentationClass/*.png')
image_se_name= [image_file.replace('\\', '/').split('/')[-1].split('.')[0] for image_file in image_SegmentationClass]


image_other=list(set(image_all_name) - set(image_se_name))
print(image_other)
for image_name in image_other:
    os.remove('data/VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(image_name))