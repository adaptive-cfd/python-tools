### LEGAL STUFF
# linspecer uses ColorBrewer,
# which has the following license:
#
#   Apache-Style Software License for ColorBrewer software and ColorBrewer Color Schemes
#
#   Copyright (c) 2002 Cynthia Brewer, Mark Harrower, and The Pennsylvania State University.
#
#   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software distributed
#   under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
#   CONDITIONS OF ANY KIND, either express or implied. See the License for the
#   specific language governing permissions and limitations under the License.
#
# The original linspecer has the following BSD license:
#
#   Copyright (c) 2015, Jonathan C. Lansey
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are
#   met:
#
#       * Redistributions of source code must retain the above copyright
#         notice, this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above copyright
#         notice, this list of conditions and the following disclaimer in
#         the documentation and/or other materials provided with the distribution
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.
### OPINION STUFF
# Lansey
# credits and where the function came from
# The colors are largely taken from:
# http://colorbrewer2.org and Cynthia Brewer, Mark Harrower and The Pennsylvania State University
#
#
# She studied this from a phsychometric perspective and crafted the colors
# beautifully.
#
# I made choices from the many there to decide the nicest once for plotting
# lines in Matlab. I also made a small change to one of the colors I
# thought was a bit too bright. In addition some interpolation is going on
# for the sequential line styles.
#
# del Rosario
# This must be a thing. How can it not.
##################################################

### CODE STUFF
import numpy as np
from scipy.interpolate import PchipInterpolator

# Interpolates an rgp colormap
def interpomap(n,cmapp):
    # Interpolate
    x    = np.linspace(1,n,len(cmapp))
    xi   = range(1,n+1)
    cmap = np.zeros((n,3))
    for ii in range(3):
        obj = PchipInterpolator(x,cmapp[:,ii])
        cmap[:,ii] = obj(xi)
    # Return flipped map
    return cmap/255.

### colorm()
# colorm returns a colormap which is really good
# for creating informative heatmap style figures.
# No particular color stands out and it doesn't
# do too badly for colorblind people either.
#
# It works by interpolating the data from the
# 'spectral' setting on http://colorbrewer2.org/ set to 11 colors
# It is modified a little to make the brightest yellow a little less bright.
def colorm(n=100):
    # Hardcoded single color
    if n == 1:
        return [0.2005, 0.5593, 0.7380];
    # Hardcoded two color
    elif n == 2:
        return [[0.2005, 0.5593, 0.7380],
                  [0.9684, 0.4799, 0.2723]];
    # Interpolate colors
    else:
        # Predefined colormap
        frac = 0.95; # Yellows out the middle color
        cmapp = np.array([[158, 1, 66],
                          [213, 62, 79],
                          [244, 109, 67],
                          [253, 174, 97],
                          [254, 224, 139],
                          [255*frac, 255*frac, 191*frac],
                          [230, 245, 152],
                          [171, 221, 164],
                          [102, 194, 165],
                          [50, 136, 189],
                          [94, 79, 162]])
        # Interpolate
        cmap = interpomap(n,cmapp)
        # Lansey returns the flipped map;
        # who am I to argue?
        return np.fliplr(cmap)

def whiteFade(n=100,thisColor='blue'):
    # Catch grey. Because it's gray.
    if thisColor == 'grey':
        thisColor = 'gray'
    # Switch
    if thisColor == 'gray':
        cmapp = np.array([[255,255,255],
                          [240,240,240],
                          [217,217,217],
                          [189,189,189],
                          [150,150,150],
                          [115,115,115],
                          [82,82,82],
                          [37,37,37],
                          [0,0,0]])
    elif thisColor == 'green':
        cmapp = np.array([[247,252,245],
                          [229,245,224],
                          [199,233,192],
                          [161,217,155],
                          [116,196,118],
                          [65,171,93],
                          [35,139,69],
                          [0,109,44],
                          [0,68,27]])
    elif thisColor =='blue':
        cmapp = np.array([[247,251,255],
                          [222,235,247],
                          [198,219,239],
                          [158,202,225],
                          [107,174,214],
                          [66,146,198],
                          [33,113,181],
                          [8,81,156],
                          [8,48,107]])
    elif thisColor == 'red':
        cmapp = np.array([[255,245,240],
                          [254,224,210],
                          [252,187,161],
                          [252,146,114],
                          [251,106,74],
                          [239,59,44],
                          [203,24,29],
                          [165,15,21],
                          [103,0,13]])
    else:
        raise ValueError("Color unrecognized")

    return interpomap(n,cmapp)

def brighten(cmap,frac=0.9):
    return [c*frac*(1.-frac) for c in cmap]

def dim(cmap,frac=0.9):
    return [c*frac for c in cmap]

def linspecer(N, colorBlindFlag=False,qualFlag=False):
    """
    ##################################################
    # by Jonathan Lansey, March 2009-2013
    # translated to Python by Zachary del Rosario, June 2016
    ##################################################
    # Usage
    #    C = linspecer(N, colorBlindFlag=False, qualFlag=False)
    #    plt.plot( X[ind], Y[ind], color=C[ind] )
    # Arguments
    #    N              = number of colors to generate
    #    colorBlindFlag = use colorblind-friendly colors
    #    qualFlag       = force qualitative graphs
    # Returns
    #    C              = array of RGB triplets
    ##################################################
    """

    # Some predefined colormaps
    set3 = np.array([[141, 211, 199],
                     [255, 237, 111],
                     [190, 186, 218],
                     [251, 128, 114],
                     [128, 177, 211],
                     [253, 180, 98],
                     [179, 222, 105],
                     [188, 128, 189],
                     [217, 217, 217],
                     [204, 235, 197],
                     [252, 205, 229],
                     [255, 255, 179]]) / 255.
    set1JL=np.array([[228, 26, 28],
                     [55, 126, 184],
                     [77, 175, 74],
                     [255, 127, 0],
                     [255*.85, 237*.85, 111*.85],
                     [166, 86, 40],
                     [247, 129, 191],
                     [153, 153, 153],
                     [152, 78, 163]]) / 255.
    set1 = np.array([[ 55*.85, 126*.85, 184*.85],
                     [228, 26, 28],
                     [ 77, 175, 74],
                     [ 255, 127, 0],
                     [ 152, 78, 163]]) / 255.
    colorBlindSet = np.array([[215,25,28],
                              [253,174,97],
                              [171*.8,217*.8,233*.8],
                              [44*.8,123*.8,182*.8]]) / 255.
    # Some tweaks
    # set3   = dim(set3,0.93)
    # set1JL = brighten(set1JL)
    # set1   = brighten(set1,0.8)

    # Colorblind mode
    if colorBlindFlag:
        if N <= 4:
            return colorBlindSet[:N]
        else:
            raise ValueError("Only 4 colorblind colors supported.")
    # Non-colorblind mode
    else:
        if N == 1:
            return np.array([55.,126.,184.])/255.
        elif N in [2,3,4,5]:
            return set1[:N]
        elif N in [6,7,8,9]:
            return set1JL[:N]
        elif N in [10,11,12]:
            # Force qualitative graphs
            if qualFlag:
                return set3[:N]
            # N=10 is (apparently) a good place to start with sequential
            else:
                return colorm(N)
        else:
            #
            return colorm(N)

# TEST
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import sin, pi
    # linspecer()!!!!!
    col1 = [linspecer(1)]
    col5 = linspecer(5)
    col9 = linspecer(9)
    col10 = linspecer(10)
    Col = [col1,col5,col9,col10]
    Num = [1,5,9,10]
    # Some synthetic data
    X = np.linspace(0,2*pi)

    for ind in range(len(Num)):
        plt.figure()
        for i in range(Num[ind]):
            plt.plot(X,sin(X*i),color=Col[ind][i])

    # Show all figures
    plt.show()
