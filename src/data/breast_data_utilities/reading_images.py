import numpy
import h5py

def read_image_binary(file_name):

    f = open(file_name, "rb")
    image = numpy.fromfile(f, dtype = 'uint16')

    height = image[0]
    width = image[1]
    image = image[2:]
    image = numpy.resize(image, (width, height)).T

    f.close()

    return image

def read_image_binary_buffer(file_name, buffer):

    f = open(file_name, "rb", buffering = 0)

    shape = numpy.frombuffer(f.read(4), dtype = numpy.uint16)

    height = numpy.int32(shape[0])
    width = numpy.int32(shape[1])

    f.readinto(buffer[0 : (height * width)])

    image = buffer[0 : (height * width)]
    image.resize((width, height))
    image = image.T

    f.close()

    return image

def read_image_mat(file_name):

    data = h5py.File(file_name, 'r')
    image = numpy.array(data['image']).T
    data.close()

    return image