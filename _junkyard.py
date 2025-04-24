import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math_utils

import argparse

def test_ft():
    """
    something is funky with the Fourier transforms so this is to figure out what
    the fftshift I think works the opposite as in MATLAB so first figure out which way they should go
    then make sure that you have the adjoint correct
    """
    data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/brain_data.mat')
    kSpace = data['d2']
    kSpace = kSpace / np.max(np.abs(kSpace))
    sMaps = data['smap']
    sMaps = sMaps / np.max(np.abs(sMaps))

    def applyF_old(x, op='notransp'):
        if op == 'transp':
            out = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        else:
            out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        return out

    def applyF(x, op='notransp'):
        if op == 'transp':
            out = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        else:
            out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x, dim=(2,3) ), norm='ortho', dim=(2,3) ), dim=(2,3) )
        return out 
    
    def applyF2_old(x, op='notransp'):
        if op == 'transp':
            out = (x.shape[2] * x.shape[3]) * torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( x, dim=(2,3) ), dim=(2,3) ), dim=(2,3) )
        else:
            out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x, dim=(2,3) ), dim=(2,3) ), dim=(2,3) )
        return out
    
    def applyF2(x, op='notransp'):
        if op == 'transp':
            out = (x.shape[2] * x.shape[3]) * torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( x, dim=(2,3) ), dim=(2,3) ), dim=(2,3) )
        else:
            out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( x , dim=(2,3) ) , dim=(2,3) ), dim=(2,3) )
        return out
    
    def applyF_noshift(x, op='notransp'):
        if op == 'transp':
            out = (x.shape[0] * x.shape[1]) * torch.fft.ifftn( x )
        else:
            out = torch.fft.fftn( x )
        return out
    
    def applyF_noshift2(x, op='notransp'):
        if op == 'transp':
            out = torch.fft.ifftn( x, norm='ortho' )
        else:
            out = torch.fft.fftn( x, norm='ortho' )
        return out
    
    def applyF_noshift3(x, op='notransp'):
        if op == 'transp':
            out = torch.fft.ifft2( x, norm='ortho' )
        else:
            out = torch.fft.fft2( x, norm='ortho' )
        return out

    kSpace_t = torch.tensor(kSpace)
    coil_8 = kSpace_t[..., -1]
    coil_8_4 = coil_8[None, None, ...]

    ft = torch.fft.ifft2(coil_8) # shifted 
    sft = torch.fft.fftshift(ft)

    ip = lambda x, y: torch.real(torch.vdot(x.flatten(), y.flatten()))

    print('test 1')
    e1 = math_utils.test_adjoint_torch(coil_8_4, applyF_old)
    print('test 2')
    e2 = math_utils.test_adjoint_torch(coil_8_4, applyF)
    print('test 3')
    e3 = math_utils.test_adjoint_torch(coil_8_4, applyF2_old)
    print('test 4')
    e4 = math_utils.test_adjoint_torch(coil_8_4, applyF2)
    print('test 5')
    e5 = math_utils.test_adjoint_torch(coil_8, applyF_noshift)
    print('test 6')
    e6 = math_utils.test_adjoint_torch(coil_8, applyF_noshift2)

    a = torch.rand(size=(32,32), dtype=torch.float64) + 1j*torch.rand(size=(32,32), dtype=torch.float64)
    print('test 7')
    e7 = math_utils.test_adjoint_torch(a, applyF_noshift)
    print('test 8')
    e8 = math_utils.test_adjoint_torch(a, applyF_noshift2)

    fft_data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/python/unrolled/fft_test.mat')
    a = fft_data['a']
    fa = fft_data['fa']
    fhfa = fft_data['fhfa']

    ta = torch.tensor(a)
    tfa = torch.tensor(fa)
    tfhfa = torch.tensor(fhfa)

    print(f'forward test: {torch.norm(applyF_noshift(ta) - tfa)}')
    # print(f'backward test: {torch.norm(applyF_noshift(applyF_noshift(ta), 'transp') - tfhfa)}')

    fft_data2 = sio.loadmat('/Users/alex/Documents/MATLAB/dataConsistency/fft_test2.mat')

    rx = torch.tensor( fft_data2['rx'] )
    ry = torch.tensor( fft_data2['ry'] )
    frx = torch.tensor( fft_data2['frx'] )
    fTry = torch.tensor( fft_data2['fTry'] )
    dp1 = fft_data2['dp1']
    dp2 = fft_data2['dp2']

    dp1_me = ip(frx, ry)
    dp2_me = ip(rx, fTry)

    print(f'ip test 1: {np.abs(dp1 - dp1_me.item())}')
    print(f'ip test 2: {np.abs(dp2 - dp2_me.item())}')

    fr1 = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( rx, dim=(0, 1) ), norm='ortho', dim=(0,1) ), dim=(0,1) )
    fr2 = torch.fft.ifftshift( torch.fft.fftn( torch.fft.fftshift( rx, dim=(0, 1) ), norm='ortho', dim=(0,1) ), dim=(0,1) )

    ifr1 = torch.fft.fftshift( torch.fft.ifftn( torch.fft.ifftshift( fr1, dim=(0, 1) ), norm='ortho', dim=(0,1) ), dim=(0,1) )
    ifr2 = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( fr1, dim=(0, 1) ), norm='ortho', dim=(0,1) ), dim=(0,1) )

    print(f'order test: {torch.norm(fr1 - fr1)}')
    print(f'backward order test: {torch.norm(ifr1 - ifr2)}')


    print('hello')

class superTest():
    def __init__(self, a):
        self.a = a
        self.asq = a**2

class subTest(superTest):
    def __init__(self, a, b):
        super().__init__(a)
        self.b = b
        self.bsq = b**2

def classTest():
    a = 15
    b = 25

    testClass = subTest(a, b)
    print(issubclass(subTest, superTest))
    print(testClass.a)
    print(testClass.asq)
    print(testClass.b)

def test_ft2():
    """
    so the rest of the code sits in a junkyard
    """

    def applyF(x, op='notransp'):
        if op == 'transp':
            out = (x.shape[0] * x.shape[1]) * torch.fft.ifftn( x )
        else:
            out = torch.fft.fftn( x )
        return out
    
    fft_data2 = sio.loadmat('/Users/alex/Documents/MATLAB/dataConsistency/fft_test2.mat')
    rx = torch.tensor( fft_data2['rx'] )

    ip = lambda x, y: torch.real(torch.vdot(x.flatten(), y.flatten()))

    err = math_utils.test_adjoint_torch(rx, applyF, ip)

    for _ in range(15):
        a = torch.randn_like(rx)
        b = torch.randn_like(rx)

        fa = applyF(a)
        fTb = applyF(b, 'transp')

        db1 = ip(a, fTb)
        db2 = ip(fa, b)

        print(f'test {torch.abs(db1 - db2)}')


def parser_test():

    # Create the parser
    parser = argparse.ArgumentParser(description='Process some file paths.')
    
    # Add arguments with defaults (you can replace these with your current hardcoded paths)
    parser.add_argument('--input', type=str, default='/default/input/path.txt',
                        help='Path to the input file')
    parser.add_argument('--output', type=str, default='/default/output/path.txt',
                        help='Path to the output file')
    parser.add_argument('--config', type=str, default='/default/config/path.json',
                        help='Path to the configuration file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Now use the paths in your code
    input_path = args.input
    output_path = args.output
    config_path = args.config
    
    # Your code here
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Config path: {config_path}")
    


if __name__ == "__main__":
    parser_test()
    # classTest()
    # test_ft()
