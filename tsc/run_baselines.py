import sys
import os

if len(sys.argv) > 1:
    GPU_ID = str(sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    
import run_fcn
import run_resnet