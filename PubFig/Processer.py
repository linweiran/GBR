

from PIL import Image
import urllib.request, io
import PIL
import os
from facenet_pytorch import MTCNN, InceptionResnetV1

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--in_dir',type=str, default='/tmp/Pubfig')
parser.add_argument('--out_dir',type=str, default='/donemodel')


args = parser.parse_args()
print("ARGS: ", args)
in_dir=args.in_dir
out_dir=args.out_dir


# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=33817 , margin=0)
'''
# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()
'''

rootdir = in_dir

if __name__ == "__main__":
	#print(rootdir)

	for dir in os.listdir(rootdir):

		f = os.path.join(rootdir, dir)
		curDir = rootdir + '/' + dir
		print(curDir)

		directory = os.fsencode(curDir)
		for file in os.listdir(directory):
				filename = os.fsdecode(file)
				#print(filename)
				fullPath = curDir + '/' + filename
				#print(fullPath)

				img = Image.open(fullPath)

				targetPath = outdir + dir + '/' + filename
				# Get cropped and prewhitened image tensor

				img_cropped = mtcnn(img, save_path= targetPath)
		
				#print(targetPath)
				#print('a')
				'''
				# Calculate embedding (unsqueeze to add batch dimension)
				img_embedding = resnet(img_cropped.unsqueeze(0))

				# Or, if using for VGGFace2 classification
				resnet.classify = True
				img_probs = resnet(img_cropped.unsqueeze(0))
				'''
	print('done')
