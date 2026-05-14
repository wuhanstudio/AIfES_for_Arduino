from building import *
import rtconfig

# get current directory
cwd     = GetCurrentDir()

# The set of source files associated with this SConscript file.
src     = Glob('src/*.c')

src    += Glob('src/basic/base/aialgo/*.c')
src    += Glob('src/basic/base/ailayer/*.c')
src    += Glob('src/basic/base/ailoss/*.c')
src    += Glob('src/basic/base/aimath/*.c')
src    += Glob('src/basic/base/aiopti/*.c')

src    += Glob('src/basic/default/ailayer/*.c')
src    += Glob('src/basic/default/ailoss/*.c')
src    += Glob('src/basic/default/aimath/*.c')
src    += Glob('src/basic/default/aiopti/*.c')

src    += Glob('src/basic/express/*.c')

path    = [cwd + '/src']
path   += [cwd + '/src/core']
path   += [cwd + '/src/basic/base']
path   += [cwd + '/src/basic/default']
path   += [cwd + '/src/basic/express']

LOCAL_CCFLAGS = ''

# XOR inference example
if GetDepend('AIFES_USING_XOR_INFERENCE_F32_EXAMPLE'):
    path   += [cwd + '/platforms/rt-thread/xor_inference_f32_example']
    src    += Glob('platforms/rt-thread/xor_inference_f32_example/*.c')

# XOR training example
if GetDepend('AIFES_USING_XOR_TRAINING_F32_EXAMPLE'):
    path   += [cwd + '/platforms/rt-thread/xor_training_f32_example']
    src    += Glob('platforms/rt-thread/xor_training_f32_example/*.c')

# CNN training example
if GetDepend('AIFES_USING_CNN_TRAINING_F32_EXAMPLE'):
    src    += Glob('src/cnn/base/ailayer/*.c')
    src    += Glob('src/cnn/base/aimath/*.c')
    src    += Glob('src/cnn/default/ailayer/*.c')
    src    += Glob('src/cnn/default/aimath/*.c')

    path   += [cwd + '/platforms/rt-thread/cnn_training_f32_example']
    src    += Glob('platforms/rt-thread/cnn_training_f32_example/*.c')

# MNIST training example
if GetDepend('AIFES_USING_MNIST_TRAINING_F32_EXAMPLE'):
    path   += [cwd + '/platforms/rt-thread/mnist_training_f32_example']
    src    += Glob('platforms/rt-thread/mnist_training_f32_example/*.c')

group = DefineGroup('aifes', src, depend = ['PKG_USING_AIFES'], CPPPATH = path, LOCAL_CCFLAGS = LOCAL_CCFLAGS)

Return('group')
