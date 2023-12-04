import tensorflow

def K_meshgrid(x, y):
    return tensorflow.meshgrid(x, y)


def K_linspace(start, stop, num):
    return tensorflow.linspace(start, stop, num)