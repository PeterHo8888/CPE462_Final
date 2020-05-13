import cpe462
import numpy as np
from PIL import Image

if __name__ == '__main__':
    a = [5, 2.5, 3]
    b = [2, 3, 4]

    c = [[1, 5, 7], [2, 6, 12], [11, 55, 23]]
    d = [[6, 12, 23], [3, 1, 6], [21, 6, 22]]
    e = [c, d]

    cpe462.hello()

    print(cpe462.dot(a, b))

    img = np.array(cpe462.load_img("a.jpg"), dtype=np.uint8)

    r = img[:,:,0].tolist()
    g = img[:,:,1].tolist()
    b = img[:,:,2].tolist()
    for i in range(100):
        np.gradient(r)

