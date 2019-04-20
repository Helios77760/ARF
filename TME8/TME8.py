import numpy as np
import matplotlib.pyplot as plt


def quantize(pixels, colors):
    newpixels = np.ndarray(pixels.shape)
    newBarycenters = np.zeros(colors.shape)
    usedColors = np.zeros(colors.shape[0])
    for i, p in enumerate(pixels):
        diff = colors - p
        sumarray = np.sum(np.multiply(diff, diff), axis=1)
        closest = np.argmin(sumarray)
        newpixels[i, :] = colors[closest, :]
        newBarycenters[closest, :] += p
        usedColors[closest] += 1

    test = 0
    for i, b in enumerate(newBarycenters):
        if usedColors[i] > 0:
            newBarycenters[i, :] = b / usedColors[i]
            test += 1
        else:
            newBarycenters[i, :] = np.random.rand(3)
    print("Nombre de couleurs utilisées : ", test)
    return newpixels, newBarycenters


def generateRandomColors(k):
    return np.random.rand(k, 3)


def reconstructionError(pixels, original):
    diff = original - pixels
    return np.mean(np.sum(np.multiply(diff, diff), axis=1))


im = plt.imread("peppers.png")[:, :, :3]
im_h, im_l, _ = im.shape
pixels = im.reshape((im_h * im_l, 3))
original = np.copy(pixels)
plt.imshow(im)
plt.show()

k = 16
colors = generateRandomColors(k)
continuer = True
while continuer:
    pixels, newColors = quantize(original, colors)
    if np.array_equal(newColors, colors):
        continuer = False
    else:
        colors = newColors
    print(reconstructionError(pixels, original))

print("Done")

plt.figure()
imnew = pixels.reshape((im_h, im_l, 3))
plt.imshow(imnew)
plt.show()
