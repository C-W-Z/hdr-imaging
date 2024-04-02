# HDR Imaging

CSIE B11902078 張宸瑋

CSIE B10902058 胡桓碩

## Dependencies

- opencv-python
- numpy
- matplotlib

## hdr.py

### Usage

```shell
$ python hdr.py -h
usage: hdr.py [-h] [-s] <input_file> <output_directory>

Read LDR images & arguments from information in <input_file> & output the HDR image 'hdr.hdr' to
<output_directory>

positional arguments:
  <input_file>        Input file (.txt) path
  <output_directory>  Output directory path

options:
  -h, --help          show this help message and exit
  -s                  Output response curve 'response-curve.png' 
                      & radiance map 'radiance-map.png' in
                      <output_directory>
```

For example:

```shell
$ cd code
$ python hdr.py ../data/test1/origin/image_list.txt ../data/test1/hdr
```

### Input File of hdr.py

For example, this is the file [data/test2/origin/image_list.txt](data/test2/origin/image_list.txt)

```txt
ALIGN=OUR      # Align images using our implementation
STD=2          # Standard image index for MTB alignment
DEPTH=3        # Recusion depth in MTB algorithm (minimum is 0)
THRESHOLD=MEAN # Threshold type in MTB algorithm
GRAYRANGE=5    # the range of intensities to ignore around threshold

LAMBDA=25      # The lambda (smoothness) used in solving response curve

# LDR filenames                 exposure
StLouisArchMultExpEV-4.72.jpeg  0.037944
StLouisArchMultExpEV-1.82.jpeg  0.283221
StLouisArchMultExpEV+1.51.jpeg  2.848100  # standard image
StLouisArchMultExpEV+4.09.jpeg  17.02992
```

Note that the `#` is just like a comment in python, we will ignore everything after `#`.

The input file should be a .txt file, for example `image_list.txt`.

There are some parameters:

`ALIGN=NONE` `ALIGN=OUR` `ALIGN=CV2`, this is alignment method, OUR is our implementation of MTB, CV2 is the OpenCV implementation of MTB.

if `ALIGN=OUR`, you need to determine the parameters below for our MTB implementation:

1. `STD` is the standard image index you chosen, if the index is not valid, we will choose the middle image as standard image.
2. `DEPTH` must >= 0, this is the recursion depth for reducing image size by half and align, default is `3`.
3. `THRESHOLD` has 3 options: `MEDIAN` `MEAN` `MEDIAN_MEAN_AVERAGE`, this is the threshold type used in MTB, default is `MEDIAN`.
5. `GRAYRANGE` must >= 0, we will ignore the intensity between `THRESHOLD-GRAYRANGE` and `THRESHOLD+GRAYRANGE` when calculate the difference between 2 bit maps.

`LAMBDA` is the smoothness used in Debevec method, default is 20.

Last, we need to write the image file names and the exposure time as follow:

```txt
# LDR filenames  exposure
your-image-1.jpg 2
your-image-2.jpg 1
your-image-3.jpg 0.5
```

The file names should not contain any spaces, and this .txt file must be in the same folder as the images you write in it.

The structure of folder will be like:

```txt
origin/
  image_list.txt
  your-image-1.jpg
  your-image-2.jpg
  your-image-3.jpg
```

## align.py

### Usage

```shell
$ python align.py -h
usage: align.py [-h] [-a] [-b] [-o <output_directory>] <input_file>

Read LDR image & arguments from information in <input_file> & output bitmaps or aligned images to
<output_directory>

positional arguments:
  <input_file>          Input file (.txt) path

options:
  -h, --help            show this help message and exit
  -a                    Output aligned images
  -b                    Output bitmaps
  -o <output_directory>
                        Output directory path, required if [-a] or [-b]
```

For example:

```shell
$ cd code
$ python align.py ../data/test1/origin/image_list.txt -a -b -o ../data/test1/align
```

Note that if neither `-a` nor `-b` is selected, no image will be saved.

### Input file for align.py

Same as input file for hdr.py, but `LAMBDA` parameter is not used.

## tonemap.py

### Usage

```shell
$ python tonemap.py -h
usage: tonemap.py [-h] <input_file> <output_directory>

Read HDR image & arguments from information in <input_file> & output the LDR images to <output_directory>

positional arguments:
  <input_file>        Input file (.txt) path
  <output_directory>  Output directory path

options:
  -h, --help          show this help message and exit
```

For example:

```shell
$ cd code
$ python tonemap.py ../data/test1/hdr/tonemap.txt ../data/test1/ldr
```

### Input file for tonemap.py

For example, this is the file [data/test1/hdr/tonemap.txt](data/test1/hdr/tonemap.txt)

```txt
FILE=hdr.hdr  # HDR filename

# Tonemapping algorithms and arguments

# Algorithm  gamma  saturation  bias  brightness
cv2_drago    2.2    1.2         0     2

# Algorithm      gamma  brightness  normalize
gamma_intensity  2.2    10          ALL
gamma_intensity  2.2    10          CHANNEL
gamma_intensity  1.5    40          ALL
gamma_intensity  1      100         ALL

# Algorithm  gamma  brightness  normalize
gamma_color  1      2.5         ALL
gamma_color  1      2.5         CHANNEL
gamma_color  2.2    1.8         ALL
gamma_color  2.2    1.8         CHANNEL

# Algorithm  a    Lwhite  delta  normalize  save-gray-image
global       2    50      1      ALL        True
global       2    50      1      CHANNEL    False
global       0.3  50      1e-6   ALL        True

# Algorithm  sigma_color  contrast  a   Lwhite  delta  normalize  save-filtered-images
bilateral    0.4          5         10  50      1      ALL        True
bilateral    0.4          5         15  50      1.2    ALL        False
bilateral    0.4          6         22  50      1.2    ALL        False
bilateral    0.8          5         15  50      1.2    ALL        False
bilateral    0.8          5         15  50      1.2    CHANNEL    False
bilateral    0.8          6         20  50      1.2    ALL        False
bilateral    2            5         10  50      1      ALL        True
```

Note that the `#` is just like a comment in python, we will ignore everything after `#`.

The input file should be a .txt file, for example `tonemap.txt`.

First you need to define a parameter `FILE`, which is the .hdr file name without spaces, and this hdr file should be in the same folder as the input file.

The folder structure will be like:

```txt
hdr/
  tonemap.txt
  hdr.hdr
```

Second, you need to write the algorithm and arguments in a line for tonemapping.

Note that the tonemap.py will run all of the lines you write, which means if you write multiple line of algorithms and arguments, tonemap.py will run tonemapping multiple times, with those algorithms and arguments.

The algorithm names and arguments are:

- `cv2_drago`:  `gamma`  `saturation`  `bias`  `brightness`
- `gamma_intensity`:  `gamma`  `brightness`  `normalize`
- `gamma_color`:  `gamma`  `brightness`  `normalize`
- `global`:  `a`  `Lwhite`  `delta`  `normalize`  `save-gray-image`
- `bilateral`:  `sigma_color`  `contrast`  `a`  `Lwhite`  `delta`  `normalize`  `save-filtered-images`

Please note that the order of parameters must be the same as above.

Parameter Explanation:

1. The most of the types of arguments are float, like `gamma` `saturation`  `bias`  `brightness`  `a`  `Lwhite`  `delta`  `sigma_color`  `contrast`

2. `save-gray-image` and `save-filtered-images` are boolean: `True` or `False`

3. `normalize` has 3 options: `NONE` `ALL` `CHANNEL`, `NONE` means tonemapping without normalization, `ALL` means normalize the whole LDR image, `CHANNEL` means normalize each channel in the LDR image individually.

4. `gamma`($\gamma$) in `gamma_intensity` and `gamma_color`: each value $x$ of hdr will be mapped to $x^{\frac{1}{\gamma}}$

5. `brightness` in `cv2_drago`, `gamma_intensity`, `gamma_color`: each value of the result of tonemapping will be multiply by `brightness`.

6. `delta` in `global` and `bilateral` is the $\delta$ in the following equation in the photographic global mapping algorithm.

$$ \bar{L}\_w = \exp \left( \frac{1}{N} \sum\_{x,y} \log \left( \delta + L_w(x,y) \right) \right) $$

7. `a` in `global` and `bilateral` is the $a$ in the following equation in the photographic global mapping algorithm.

$$ L_m(x,y) = \frac{a}{\bar{L}_w} L_w(x,y) $$

8. `Lwhite` in `global` and `bilateral` is the $L_{white}$ in the following equation in the photographic global mapping algorithm.

$$ L_d(x,y) = \frac{ L_m(x,y) \left( 1 + \frac{ L_m(x,y) }{ L^2_{white}(x,y) } \right) }{ 1 + Lm(x,y) } $$

9. `sigma_color` in `bilateral` is the $\sigma$ parameter in the Guassion kernel of intensity, in bilateral filter. And the $\sigma$ parameter in the Guassion kernel of distance is auto-determined.

10. `contrast` in `bilateral` is the value that determines how much the low frequency part (base) filtered by bilateral filter is compressed.
