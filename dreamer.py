#!/usr/bin/python
__author__ = 'samim'

# Imports
import argparse
import time
import os
import errno
import subprocess

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format
import caffe

#Loading DNN model
model_name = 'bvlc_googlenet'
#model_name = 'age_net'
# model_name = 'gender_net'
# model_name = 'hybridCNN+'
# model_name = 'placesCNN+'
# model_name = 'vggf'
# model_name = 'flowers'
model_path = '/Users/samim/caffe/models/' + model_name + '/'
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'net.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    #display(Image(data=f.getvalue()))

def blur(img, sigma):
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

# First we implement a basic gradient ascent step function, applying the first two tricks // 32:
def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True): 
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    # ox, oy = np.random.normal(0, max(1, jitter), 2).astype(int) # use gaussian distribution    

    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
	# src.data[0] += np.random.normal(0, 1, (3, 224, 224)) # add some noise

    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]

    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)  


# Next we implement an ascent through different scales. We call these scales "octaves".
def deepdream(net, base_img, iter_n=10, octave_n=4, step_size=1.5, octave_scale=1.4, jitter=32, end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end,step_size=step_size, jitter=jitter, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            #showarray(vis)
            print octave, i, end, vis.shape
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])



# Animaton functions

def resizePicture(image,width):
	img = PIL.Image.open(image)
	basewidth = width
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	return img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)

def morphPicture(filename1,filename2,blend,width):
	img1 = PIL.Image.open(filename1)
	img2 = PIL.Image.open(filename2)
	if width is not 0:
		img2 = resizePicture(filename2,width)
	return PIL.Image.blend(img1, img2, blend)

def make_sure_path_exists(path):
    # make sure input and output directory exist, if not create them. If another error (permission denied) throw an error.
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def main(inputdir,outputdir,preview,octaves,octave_scale,iterations,jitter,zoom,stepsize,blend,layers):
	make_sure_path_exists(inputdir)
	make_sure_path_exists(outputdir)

	if preview is None: preview = 0
	if octaves is None: octaves = 4
	if octave_scale is None: octave_scale = 1.5
	if iterations is None: iterations = 10
	if jitter is None: jitter = 32
	if jitter is None: jitter = 32
	if zoom is None: zoom = 1
	if stepsize is None: stepsize = 1.5
	if blend is None: blend = 1.5
	if layers is None: layers = ['inception_4c/output']

	var_counter = 1
	vidinput = os.listdir(inputdir)
	vids = [];

	for frame in vidinput:
		if not ".jpeg" in frame: continue
		vids.append(frame)

	img = PIL.Image.open(inputdir+'/'+vids[0])
	if preview is not 0:
		img = resizePicture(inputdir+'/'+vids[0],preview)
	
	frame = np.float32(img)

	for v in range(len(vids)):
		vid = vids[v]

		now = time.time()
		#net.blobs.keys()

		h, w = frame.shape[:2]
		s = 0.05 # scale coefficient

		for i in xrange(zoom):
			print 'Processing: ' + inputdir+'/'+ vid
			
			endparam = layers[var_counter % len(layers)]
			var_end = endparam.replace("/", "-");
			
			frame = deepdream(net, frame, iter_n=iterations,step_size=stepsize, octave_n=octaves, octave_scale=octave_scale, jitter=jitter, end=endparam)

			later = time.time()
			difference = int(later - now)
			filenameCounter = 10000 + var_counter
			saveframe = outputdir+"/"+str(filenameCounter) + '_' + "_octaves"+str(octaves)+"_iterations"+str(iterations)+"_octavescale"+str(octave_scale)+'_net'+var_end+ '_jitter' +  str(jitter) + '_stepsize' + str(stepsize) + '_blend' + str(blend) + '_renderTime' + str(difference) + 's' + '_filename'+ vid 

			# Stats
			print '***************************************'
			print 'Saving Image As: ' + saveframe
			print 'Frame ' + str(var_counter) + ' of ' + str(len(vids))
			print 'Frame Time: ' + str(difference) + 's' 
			timeleft = difference * (len(vids) - var_counter)
			m, s = divmod(timeleft, 60)
			h, m = divmod(m, 60)
			print 'Estimated Total Time Remaining: ' + str(timeleft) + 's (' + "%d:%02d:%02d" % (h, m, s) + ')' 
			print '***************************************'

			PIL.Image.fromarray(np.uint8(frame)).save(saveframe)

			newframe = inputdir + '/' + vids[v+1]
			frame = morphPicture(saveframe,newframe,blend,preview)
			frame = np.float32(frame)

			var_counter += 1

def extractVideo(inputdir, outputdir):
	print subprocess.Popen('ffmpeg -i '+inputdir+' -f image2 '+outputdir+'/image-%3d.jpeg', shell=True, stdout=subprocess.PIPE).stdout.read()

def createVideo(inputdir,outputdir,framerate):
	print subprocess.Popen('ffmpeg -r '+str(framerate)+' -f image2 -pattern_type glob -i "'+inputdir+'/*.jpeg" '+outputdir, shell=True, stdout=subprocess.PIPE).stdout.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepDreamAnim')
    parser.add_argument('-i','--input', help='Input directory',required=True)
    parser.add_argument('-o','--output',help='Output directory', required=True)
    parser.add_argument('-p','--preview',help='Preview image width. Default: 0', type=int, required=False)
    parser.add_argument('-oct','--octaves',help='Octaves. Default: 4', type=int, required=False)
    parser.add_argument('-octs','--octavescale',help='Octave Scale. Default: 1.4', type=float, required=False)
    parser.add_argument('-itr','--iterations',help='Iterations. Default: 10', type=int, required=False)
    parser.add_argument('-j','--jitter',help='Jitter. Default: 32', type=int, required=False)
    parser.add_argument('-z','--zoom',help='Zoom in Amount. Default: 1', type=int, required=False)
    parser.add_argument('-s','--stepsize',help='Step Size. Default: 1.5', type=float, required=False)
    parser.add_argument('-b','--blend',help='Blend Amount. Default: 0.5', type=float, required=False)
    parser.add_argument('-l','--layers',help='Layers Loop. Default: inception_4c/output', nargs="+", type=str, required=False)
    parser.add_argument('-e','--extract',help='Extract Frames From Video.', type=int, required=False)
    parser.add_argument('-c','--create',help='Create Video From Frames.', type=int, required=False)

    args = parser.parse_args()

    if args.extract is 1:
    	extractVideo(args.input, args.output)
    elif args.create is 1:
    	createVideo(args.input, args.output,24)
    else:
    	main(args.input, args.output, args.preview, args.octaves, args.octavescale, args.iterations, args.jitter, args.zoom, args.stepsize, args.blend, args.layers)
