import random
import rootutils
rootutils.setup_root(search_from=__file__, indicator='setup.py', pythonpath=True)
# src/data/components/aug/wrapper_v2.py
# from draw import *
from src.data.components.aug.augment_v2 import *
from src.data.components.aug.base import thinning
import os 
import json

random.seed(None)

MORPH_WINDOW_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
MORPH_WINDOW_ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
class Augmenter:
    def __init__(self, 
                texture_path="", 
                bg_checkpoint="", 
                task="train",  
                cnts_checkpoint=None):
        self.texture = None 
        self.bg_checkpoint = bg_checkpoint
        self.task = task
        # self.cnts_checkpoint = cnts_checkpoint + "{}.npz"

        # Since line pattern is gone, we gonna use this as alternative background
        self.load_texture(texture_path)
        
        # if Augmenter.texture is None:
        #     img = cv2.imread(texture_path)
        #     if img is None:
        #         print("Cant read image")
        #     bg_color = np.mean(img, axis=(0, 1))
        #     threshold = np.max((img < bg_color).astype(np.uint8), axis = 2)
        #     cropped = crop_img(img / 255, threshold, 0, 0, 0, 0, bg_color=(1, 1, 1))[0]
        #     cropped = cropped
        #     Augmenter.texture = cropped.copy()
        
        # self.cnts_checkpoint = cnts_checkpoint + "{}.npz"
        # self.bg_checkpoint = bg_checkpoint + "{}.npz"
    
    def load_texture(self, path):
        if os.path.isdir(path):
            self.texture = []
            for fname in os.listdir(path):
                texture_path = os.path.join(path, fname)
                assert os.path.isfile(texture_path)
                
                # Is it all BGR or maybe RGB?
                self.texture.append(cv2.imread(texture_path))
        else:
            self.texture = cv2.imread(path)

    def bake_bg(self, image, path=None):
        data = None
        pwd = os.path.join(self.bg_checkpoint, self.task)
        # Check background folder availability
        # print(os.path.exists(pwd))
        create = False
        parts = path.split("/")
        # Sequentially reaching the file
        for part in parts:
            pwd = os.path.join(pwd, part)
            if '.' in part:
                # print(pwd)
                if os.path.exists(pwd):
                    print(pwd, 'cached')
                    data = np.load(pwd, allow_pickle=True)
                else: 
                    print(pwd, 'saving')
                    thresh, bg_color, bg, freq = extract_background(image.copy())
                    np.savez(pwd, thresh=thresh, bg_color=bg_color, bg=bg, freq=freq)
            else:
                if not os.path.exists(pwd):
                    os.mkdir(pwd)
                    create = True
        
        # data = np.load(self.bg_checkpoint.format(path))
        # extract data 
        thresh, bg_color, bg, freq = extract_background(image.copy())
        
        return {"thresh": thresh, "bg_color": bg_color, "bg": bg, "freq": freq} if data is None else data 

    # use for sequential data only...
    @staticmethod
    def translate(label):
        output = ""
        num_punc = 0
        for char in label:
            output += Augmenter.translator[char]
        
        for char in output:
            if char == '^' or char == '~' or char == '`' or char == '\'' or char == '?' or char == '.' or char == '_':
                num_punc += 1
        
        return output, len(output) - num_punc, num_punc

    @staticmethod
    def randomRange(range):
        if len(range) == 1:
            return range[0]
        else:
            return random.random() * (range[1] - range[0]) + range[0]
    
    @staticmethod
    def randomPick(select, p):
        return np.random.choice(select, p)

    def transform_img(self, img,  
                            mask=None,
                            rotate=(-15, 15), 
                            shear_x=(-0.1, 0.1), 
                            shear_y=(-0.1, 0.1), 
                            opacity=(0.6, 1),
                            scale=(0.5, 1.2),
                            bx=(2, 3),
                            by=(0, 6),
                            ex=(0, 5),
                            ey=(0, 5),
                            logger=None,
                            debug= None,
                            keep_mask=False,
                            export=False,
                            fname=None,
                            borderMode='constant',
                            morph_size=(-2, 2.5),
                            morph=False):
        
        # Arguments configuration
        rotate = int(Augmenter.randomRange(rotate))
        shear_x = Augmenter.randomRange(shear_x)
        shear_y = Augmenter.randomRange(shear_y)
        alpha = borderMode != 'constant' 
        # print(alpha)

        scale = Augmenter.randomRange(scale)
        # Recenter image
        bx = int(Augmenter.randomRange(bx)) + int(max(1 - scale, 0) * img.shape[1] / 2)
        by = int(Augmenter.randomRange(by)) + int(max(1 - scale, 0) * img.shape[1] / 2)
        ex = int(Augmenter.randomRange(ex)) + int(max(1 - scale, 0) * img.shape[0] / 2)
        ey = int(Augmenter.randomRange(ey)) + int(max(1 - scale, 0) * img.shape[0] / 2)
        print(fname)
        # Load background details
        if fname is not None:
            # exclude file extension
            
            fname = fname.split(".")[0] + ".npz"
            # read data
            data = self.bake_bg(img, path=fname)
            thresh = data['thresh']
            bg_color = data['bg_color'].tolist()
            bg = data['bg']
            freq = data['freq']
            bg_color.append(0)  
        else:
            data = self.bake_bg(img, path=fname)

        # Mask extraction
        if mask is None:
            mask = sharp_mask(img, 0.95, 0.95)
        else:
            mask = mask

        # Configure background
        if borderMode == 'replicate':
            bg_color = None
        
        # Transform image
        transformed, transformed_mask, transform_matrix, alpha_mask = warp_transform(img, 
                                                                                    mask=mask, 
                                                                                    scale=scale,
                                                                                    angle=rotate, 
                                                                                    shear_x=shear_x, 
                                                                                    shear_y=shear_y, 
                                                                                    translate_x=bx,
                                                                                    translate_y=by,
                                                                                    pad_x=ex,
                                                                                    pad_y=ey,
                                                                                    bg_color=bg_color, 
                                                                                    export=export,
                                                                                    alpha=alpha)
        
        # Crop content and padding
        points = np.array(np.where(transformed_mask > 0)).T
        y, x, h, w = cv2.boundingRect(points)
        cropped = transformed[max(0,y-by):min(y+h+ey, transformed.shape[0]),max(0,x-bx):min(x+w+ex,transformed.shape[1])]
        cropped_mask = transformed_mask[max(0,y-by):min(y+h+ey, transformed.shape[0]),max(0,x-bx):min(x+w+ex,transformed.shape[1])]
        
        # print("Bounding rect: x {0}-{1}, y {2}-{3}".format(max(0, x-bx), x + w + ex, max(0, y - ey), y + h + ey))
        # print("Original img shape:{}".format(transformed.shape))
        # print("Cropped img shape:{}".format(cropped.shape))
        # print(alpha_mask.shape)
        # if alpha_mask is not None:
        #     alpha_mask = np.ones(img.shape[:2])
        # Fill background
        if alpha_mask is not None:
            # print("Apply native")
            alpha_mask = alpha_mask[max(0,y-by):min(y+h+ey, transformed.shape[0]),
                                    max(0,x-bx):min(x+w+ex, transformed.shape[1])]
            if morph is True:
                morph_size = int(Augmenter.randomRange(morph_size) * scale) 
                # print(morph_size)
                if morph_size > 0: 
                    # print(morph_size)
                    cropped_mask = thinning(cropped_mask)
                    cropped_mask = cv2.dilate(cropped_mask, MORPH_WINDOW_CROSS, iterations=morph_size)
                    alpha_mask[cropped_mask == 0] = 0
                elif morph_size < 0:
                    cropped_mask = thinning(cropped_mask)
                    cropped_mask = cv2.dilate(cropped_mask, MORPH_WINDOW_ELLIPSE, iterations=int(-morph_size * 1.5))
                    thicken_mask = np.stack([cropped_mask] * 3, axis = 2).astype(float)
                    thicken_mask = cv2.GaussianBlur(thicken_mask, (3, 3), 0)
            # print(alpha_mask.shape, cropped.shape)
            
            # cropped, alpha_mask = elasticdeform.deform_random_grid([cropped, alpha_mask], 
            #                                                         sigma=np.mean(cropped.shape) / 4, 
            #                                                         points=3, 
            #                                                         axis=(0, 1))
            # Initialize background seed.
            randomnizer = np.random.default_rng()
            alpha_mask = cv2.GaussianBlur(alpha_mask, (3, 3), 0)
            # print(np.sum((alpha_mask > 0).astype(int)), alpha_mask.shape)
            alpha_mask = np.stack((alpha_mask, alpha_mask, alpha_mask), axis=2)
            # index_mask = np.random.choice(range(freq.shape[0]), cropped.shape[0] * cropped.shape[1], p=freq)
            # background = np.array([bg[index] for index in index_mask], dtype=float).reshape(cropped.shape)
            
            # Initialize background 
            if np.max(freq) < 0.7: 
                background = randomnizer.choice(bg, 
                                                (cropped.shape[0], cropped.shape[1]), 
                                                axis=0, 
                                                p=freq).astype(float)
            else:
                # print("Background is noiseless, make synthetic background")
                assert self.texture is not None
                if isinstance(self.texture, list):
                    pos = np.random.randint(0, len(self.texture))
                    texture = self.texture[pos]
                else:
                    texture = self.texture
                y, x = randomnizer.integers(0, texture.shape[0] - cropped.shape[0]), randomnizer.integers(0, texture.shape[1] - cropped.shape[1])
                background = cv2.resize(cv2.GaussianBlur(texture, (3, 3), 0), 
                                        [cropped.shape[1], cropped.shape[0]],
                                        interpolation = cv2.INTER_LINEAR).astype(float)
            
            # Matching background & foreground color
            ratio = np.mean(background, axis=(0, 1)) / bg_color[:3]
            background /= ratio
            # print("Background color profile: {}".format(np.mean(background, axis=[0, 1])))
            # Fusing image
            background = cv2.GaussianBlur(background, (3, 3), 0)
            alpha_mask *= Augmenter.randomRange(opacity)
            # check validity
            # print(np.median(alpha_mask))
            output = background * (1 - alpha_mask) + cropped * alpha_mask
            
            if morph is True and morph_size < 0:
                output *= 1 - thicken_mask * 0.8
            output = np.clip(output, 0, 255)
            output = output.astype(int).astype(np.uint8)
        else:
            output = cropped
        
        if isinstance(debug, str):
            fname = fname.split("/")[-1].split(".")[0]
            # print("Logging output for debugging")

            cv2.imwrite(os.path.join(debug, "original_{}.jpg").format(fname), img)
            # content mask
            log_mask = (np.stack([mask] * 3, axis=2) * 255).astype(np.uint8)
            log_t_mask = (np.stack([transformed_mask] * 3, axis=2) * 255).astype(np.uint8)
            log_c_mask = (np.stack([cropped_mask] * 3, axis=2) * 255).astype(np.uint8)
            # log_alpha = (np.stack([alpha_mask[:, :, np.newaxis]] * 3, axis=2) * 255).astype(np.uint8) 
            
            cv2.imwrite(os.path.join(debug, "mask_{}.jpg").format(fname), log_mask)
            cv2.imwrite(os.path.join(debug, "transformed_mask_{}.jpg").format(fname), log_t_mask)
            cv2.imwrite(os.path.join(debug, "transformed_{}.jpg").format(fname), transformed) 
            cv2.imwrite(os.path.join(debug, "alpha_mask_{}.jpg").format(fname), alpha_mask * 255)
            cv2.imwrite(os.path.join(debug, "background_{}.jpg").format(fname), background)
            # cv2.imwrite(os.path.join(debug, "deformed_{}.jpg").format(fname), transformed)
            cv2.imwrite(os.path.join(debug, "cropped_{}.jpg").format(fname), cropped)
            cv2.imwrite(os.path.join(debug, "cropped_mask_{}.jpg").format(fname), log_c_mask)
            cv2.imwrite(os.path.join(debug, "background_alpha_{}.jpg").format(fname), background * (1 - alpha_mask))
            cv2.imwrite(os.path.join(debug, "content_alpha_{}.jpg").format(fname), cropped * alpha_mask)
        return output.astype(np.uint8), cropped_mask, transform_matrix 
            
        
        # return augment_img(img, rotate=rotate,
        #                         shear_x=shear_x,
        #                         shear_y=shear_y,
        #                         logger=logger,
        #                         debug=debug,
        #                         keep_mask=keep_mask,
        #                         export=export)
    

    @staticmethod
    def add_noise(img,  label,
                        mask=None,
                        transform=None,
                        bg_color=1,
                        line_width=(2, 5),
                        spacing=(1, 6),
                        y_noise=(0, 5),
                        skew_noise=(0, 0.1),
                        intent=(0.8, 0.1),
                        align=(-1.2, 0.7),
                        ignore_skew=(0.8, 0.2),
                        noise_level=(0, 0.1),
                        axis = (0.2, 0.8),
                        dark = (0.8, 1.2),
                        fname=None,
                        num_line=(1, 2),
                        distort_level=(0, 5),
                        distort_degree=(0.5, 0.5)):
        """ Adding noise and patterns into image, ignore if text doesn't follow Latin style.
        """
        line_width = int(Augmenter.randomRange(line_width))
        num_line = int(Augmenter.randomRange(num_line))
        dark = Augmenter.randomRange(dark)

        if fname is None:
            print("File not found" + self.cnts_checkpoint.format(fname))
            cnts, boxes, is_char, centroid = find_text_and_punc(img, mask, label)
            transform = None
        else:
            # print("Load parameter")
            param = np.load(self.cnts_checkpoint.format(fname), allow_pickle=True)
            boxes = param['boxes']
            is_char = param['is_char']
            centroid = param['centroid']
            # print(is_char)
        
        # print(line_width)
        pattern = cv2.resize(Augmenter.texture, (line_width, line_width), interpolation=cv2.INTER_LINEAR)
        padding = (np.array(pattern.shape, dtype=int) / 3).astype(int)
        pattern = cv2.copyMakeBorder(pattern, padding[0], padding[0], padding[1], padding[1], borderType=cv2.BORDER_CONSTANT, value=(1, 1, 1))
        pattern = cv2.blur(pattern, (2, 2))
        pattern[ pattern < 1] /= dark
        
        output = img.copy()
        for i in range(num_line):
            i_y_noise = Augmenter.randomRange(y_noise)
            i_skew_noise = Augmenter.randomRange(skew_noise)
            i_intent = np.random.choice([False, True], p=intent)
            i_align = Augmenter.randomRange(align)
            i_ignore_skew = np.random.choice([False, True], p=ignore_skew)
            i_noise_level = Augmenter.randomRange(noise_level)
            i_axis = np.random.choice([0, 1], p=axis)
            i_spacing = max(Augmenter.randomRange(spacing), - line_width + 1)
            a, b = line_vector(boxes, 
                            is_char, 
                            centroid, 
                            img.shape, 
                            transform=transform,
                            y_noise=i_y_noise, 
                            skew_noise=i_skew_noise, 
                            intent=i_intent, 
                            align=i_align, 
                            ignore_skew=i_ignore_skew, 
                            axis=i_axis)
            
            output = draw_line(output, pattern, i_spacing, a, b, axis=i_axis, noise_level=noise_level)
        
                
        output = elasticdeform.deform_random_grid(output, 
                                                  sigma=Augmenter.randomRange(distort_level), 
                                                  points=np.random.choice([3, 4], p=distort_degree), 
                                                  axis=(0, 1), 
                                                  mode='nearest')
        return output

    def full_augment(self, 
                    img, 
                    choice=(0.6, 0.2, 0.2), 
                    fname=None, 
                    borderMode='native'):
        pose = np.random.choice((1, 2, 3), p=choice)
        
        if pose == 1:
            return self.transform_img( img, 
                                            fname=fname, 
                                            scale=(1., ),
                                            rotate=(0, ), 
                                            shear_x=(0, ), 
                                            shear_y=(0, ), 
                                            opacity=(1, ),
                                            borderMode=borderMode)[0]
        elif pose == 2:
            return self.transform_img( img, 
                                            fname=fname,
                                            scale=(1., ),  
                                            rotate=(0, ), 
                                            shear_x=(0, ), 
                                            shear_y=(0, ), 
                                            morph_size=(-3, 3.5),
                                            morph=True,
                                            borderMode=borderMode)[0]
        elif pose == 3:
            return self.transform_img( img, 
                                            fname=fname,
                                            morph_size=(-3, 3.5),
                                            morph=True,
                                            borderMode=borderMode)[0]

    @staticmethod
    def process(img, 
                label, 
                sample, 
                p=None, 
                fname=None, 
                borderMode='constant'):
        output = []
        for i in range(sample):
            if p is None:
                output.append(Augmenter.full_augment(img, label, fname=fname, borderMode=borderMode))
            else:
                output.append(Augmenter.full_augment(img, label, choice=p, fname=fname, borderMode=borderMode))

        return output 
    

if __name__ == "abc":
    augmenter = Augmenter("./potato/asset/translate.txt", "./potato/asset/texture.png", "./potato/asset/line/", "./potato/asset/background/")
    img_dir = "/work/hpc/firedogs/data_/new_train/"
    img_label = "/work/hpc/firedogs/data_/train_gt.txt"
    with open(img_label, "r") as file:
        for line in file:
            parts = line.strip().split()
            img = cv2.imread(img_dir + parts[0])
            print(parts[0])
            filename = parts[0].split(".")[0]
            output = augmenter.process(img, parts[1], 1, p=(0, 0, 1, 0), fname=filename, borderMode='native')
            for i in range(len(output)):
                cv2.imwrite("/work/hpc/firedogs/potato/output/augmented_{}_{}".format(i, parts[0]), output[i])

if __name__ =="__main__":
    augmenter = Augmenter(  texture_path="/data/hpc/potato/sinonom/data/augment/background/base/", 
                            bg_checkpoint="/data/hpc/potato/sinonom/data/augment/background/",
                            task="train")
    img_dir = "/data/hpc/potato/sinonom/data/wb_recognition_dataset/train/"
    # /data/hpc/potato/sinonom/data/wb_recognition_dataset/train/1/nlvnpf-0137-01-022_crop_23.jpg
    # filename = "0/12_0.png"
    # label, fname = filename.split("/")
    # img = cv2.imread(img_dir + filename)
    # label = int(label)
    # transformed, _, _ = augmenter.transform_img(img, 
    #                                             fname=filename,
    #                                             debug="/data/hpc/potato/sinonom/data/debug", 
    #                                             borderMode='native',
    #                                             scale=(0.6, ),
    #                                             morph_size=(-3.,),
    #                                             morph=True)
    # img_label = "/work/hpc/firedogs/data_/train_gt.txt"
    # parts = ["train_img_88652.png", "xeva"]
    # img = cv2.imread(img_dir + parts[0])
    # print(parts[0])
    # filename = parts[0].split(".")[0]
    # processed = augmenter.process(img, 
    #                               parts[1], 
    #                               1, 
    #                               p=(0, 1, 0, 0), 
    #                               fname=filename,
    #                               borderMode='replicate')[0]
    # # print(processed.shape, processed.dtype)
    # cv2.imwrite("/data/hpc/potato/sinonom/data/debug/{}.jpg".format(fname.split(".")[0]), transformed)
    with open("/data/hpc/potato/sinonom/data/wb_recognition_dataset/manifest_split.json", "r") as file: 
        dataset = json.load(file)['train']
    for key in dataset.keys(): 
        samples = dataset[key]
        for sample in samples:
            img = cv2.imread(img_dir + sample)
            # print(sample)
            augmenter.bake_bg(img, path=sample)

        