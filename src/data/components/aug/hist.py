import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from deaug import *
import matplotlib 
matplotlib.use('Agg')


def figure_img(valuess):
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    # ax.invert_yaxis()
    for values in valuess:
        ax.plot(range(values.shape[0]), values)
    # ax.axis('off')
    canvas.draw()
    # print(canvas.print_to_buffer())
    flattened = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    return flattened.reshape(*canvas.get_width_height()[::-1], -1)[:, :, :3]

def concat(imgs, axis=0):
    # imgs[1] = cv2.resize(imgs[1], *reversed(imgs[0].shape))
    shapes = np.array([img.shape[:2] for img in imgs])
    # print(shapes)
    if axis == 0:
        cum = 0
        concat = np.zeros((np.sum(shapes[:, 0]), np.max(shapes[:, 1]), 3))
        for i in range(shapes.shape[0]):
            concat[cum:cum+shapes[i, 0], :shapes[i, 1]] = imgs[i]
            cum += shapes[i, 0]
    else:
        cum = 0
        concat = np.zeros((np.max(shapes[:, 0]), np.sum(shapes[:, 1]), 3))
        for i in range(shapes.shape[0]):
            concat[:shapes[i, 0], cum: cum + shapes[i, 1]] = imgs[i]
            cum += shapes[i, 1]
            
    return concat

if __name__ == "__main__":
    # /work/hpc/firedogs/BKAI_private/data_private/private_test_1.jpg
    ref_dir = "/work/hpc/firedogs/BKAI_private/data_private_old/"
    img_dir = "/work/hpc/firedogs/potato/public_test_optim/check/{}"
    export_dir = "/work/hpc/firedogs/potato/public_test_optim/images/{}"
    fname_file = "/work/hpc/firedogs/potato/public_test_optim/fix.txt"
    debug_dir = "/work/hpc/firedogs/potato/public_test_optim/private_hist/{}_2_hist.jpg"
    # fnames = np.loadtxt(fname_file, dtype=str)
    # fnames = np.unique(fnames)
    test_name = "private_test_{0}.{1}"
    tail = ["jpg", "png"]
    # if args
    i = 0
    penalty = 0
    while True:
        i += 1
        fname = test_name.format(i, tail[0])
        
        if penalty > 500: 
            break
        # print(ref)
        if os.path.isfile(ref_dir.format(fname)) is False:
            fname = test_name.format(i, tail[1])
            
        if os.path.isfile(ref_dir.format(fname)) is False:
            print("/work/hpc/firedogs/BKAI_private/data_private/private_test_1.jpg")
            print(ref_dir.format(fname))
            # print("Done")
            penalty += 1
            continue
        
        filename = fname.split(".")[0]
        if os.path.isfile(debug_dir.format(filename)) is True:
            print("Skip", fname)
            continue
            # pass
        print(fname)
        img = cv2.imread(ref_dir.format(fname), cv2.IMREAD_GRAYSCALE)
        # mask = sharp_mask(img, 0.8, 0.8)
        # hist = np.sum(mask, axis=1)
        # print(hist)
        plot = figure_img([np.sum(img, axis=1), np.sum(img, axis=0)])
        output = concat([np.stack((img, img, img), axis=2), plot], axis= 1)
        cv2.imwrite(debug_dir.format(filename), output)
        