import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
import matplotlib.pyplot as plt
import socket


figure_no = 0

if socket.gethostname() == 'zliao-AIML':
    matplotlib.use('TkAgg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def set_learning_rate(optimizer, epoch_no, args):
    if args.lr_decay_policy == 'exp':
        import math
        lr_decay_rate = math.exp((math.log(args.learning_rate_end) - math.log(args.learning_rate)) / (args.num_epochs - 1))
        epoch_lr = args.learning_rate * lr_decay_rate**epoch_no
    elif args.lr_decay_policy == 'linear':
        lr_decay_rate = (args.learning_rate - args.learning_rate_end) / (args.num_epochs - 1)
        epoch_lr = args.learning_rate - lr_decay_rate * epoch_no
    else:
        raise ValueError('Unsupported learning rate policy.')

    print('Using learning rate: {:0.5f}'.format(epoch_lr))
    for pg in optimizer.param_groups:
        pg['lr'] = epoch_lr


def draw(mat, img_no=0):
    global figure_no

    if str(type(mat)) == "<class 'torch.Tensor'>":
        mat = mat.detach().cpu().numpy()

    plt.figure(num=figure_no)

    for row in mat:
        plt.plot(row)


def plot(img, img_no=0, channel_no=0, normalize=False, title=None):
    global figure_no
    if str(type(img)) == "<class 'torch.Tensor'>":
        img = img.detach().cpu().numpy()

    if len(img.shape) == 4:
        print('plot using img no: {}'.format(img_no))
        img = img[img_no]

    if img.shape[0] == 1 and len(img.shape) == 3:
        img = np.squeeze(img)
    elif img.shape[0] == 3 and len(img.shape) == 3:
        img = np.moveaxis(img, (0, 1, 2), (2, 0, 1))
    elif img.shape[-1] == 3 and len(img.shape) == 3:
        img = img
    elif len(img.shape) == 3:
        img = img[channel_no]

    # if len(np.unique(img)) < 256:
    #     if np.max(img) < 1:
    #         img = img*255
    #
    #     img = img.astype(np.uint8)

    if len(img.shape) == 2:
        fig=plt.figure(num=figure_no)
        plt.imshow(img, cmap='gray')
        fig.suptitle(title)
    else:
        plt.figure(num=figure_no)
        plt.imshow(img)
        # fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True, num=figure_no)
        # fig.suptitle(title)
        # if normalize:
        #     img = img - np.min(img, axis=(1, 2), keepdims=True)
        #     img = img / np.max(img, axis=(1, 2), keepdims=True)
        #
        # ax[0, 0].imshow(img)
        # ax[0, 0].set_title("All channels")
        #
        # img_r = img.copy() * 0 + img.min()
        # img_r[:, :, 0] = img[:, :, 0]
        # ax[0, 1].imshow(img_r)
        # ax[0, 1].set_title('R')
        #
        # img_g = img.copy() * 0 + img.min()
        # img_g[:, :, 1] = img[:, :, 1]
        # ax[1, 0].imshow(img_g)
        # ax[1, 0].set_title('G')
        #
        # img_b = img.copy() * 0 + img.min()
        # img_b[:, :, 2] = img[:, :, 2]
        # ax[1, 1].imshow(img_b)
        # ax[1, 1].set_title('B')

    plt.pause(0.5)
    # plt.show()
    figure_no += 1


def augment_segmentations(source_images, segs, labels):

    augmented_images = list()
    for im_idx, sim in enumerate(source_images):
        # the image
        sim = sim.numpy()
        sim = np.moveaxis(sim, (0, 1, 2), (2, 0, 1))

        # attention map
        seg_maps = segs[im_idx].detach().cpu().numpy()
        target_maps = labels[im_idx].detach().cpu().numpy()
        for cl in range(min(seg_maps.shape[0], 4)):
            seg_map = seg_maps[cl]
            seg_map = (seg_map >= 0.5).astype(np.float)
            target_map = target_maps[cl]

            # to uint8
            sim_uint8 = (sim * 255).astype(np.uint8)

            # for target_map
            att_map_uint8 = (target_map * 255).astype(np.uint8)
            att_map_uint8 = cv2.applyColorMap(att_map_uint8, cv2.COLORMAP_JET)

            augmented = cv2.addWeighted(sim_uint8, 1.0, att_map_uint8, 0.5, 0.0)
            augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
            augmented = np.moveaxis(augmented, (0, 1, 2), (1, 2, 0)) / 255.
            augmented_images.append(torch.tensor(augmented))

            # for seg_map
            att_map_uint8 = (seg_map * 255).astype(np.uint8)
            att_map_uint8 = cv2.applyColorMap(att_map_uint8, cv2.COLORMAP_JET)

            augmented = cv2.addWeighted(sim_uint8, 1.0, att_map_uint8, 0.5, 0.0)
            augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
            augmented = np.moveaxis(augmented, (0, 1, 2), (1, 2, 0)) / 255.
            augmented_images.append(torch.tensor(augmented))

    return augmented_images


def generate_attention_overlay_images(source_images, atts):

    augmented_images = list()
    for im_idx, sim in enumerate(source_images):
        # the image
        sim = sim.numpy()
        sim = np.moveaxis(sim, (0, 1, 2), (2, 0, 1))

        # attention map
        att_maps = atts[im_idx].detach().cpu().numpy()
        for cl in range(min(att_maps.shape[0], 5)):
            att_map = att_maps[cl]
            att_map = cv2.resize(att_map, source_images.shape[2:], interpolation=cv2.INTER_CUBIC)

            min_value = np.min(np.min(att_map, axis=0, keepdims=True), axis=1, keepdims=True)
            max_value = np.max(np.max(att_map, axis=0, keepdims=True), axis=1, keepdims=True)
            att_map = (att_map - min_value) / (max_value - min_value)

            # to uint8
            sim_uint8 = (sim * 255).astype(np.uint8)

            # for seg_map
            att_map_uint8 = (att_map * 255).astype(np.uint8)
            att_map_uint8 = cv2.applyColorMap(att_map_uint8, cv2.COLORMAP_JET)

            augmented = cv2.addWeighted(sim_uint8, 0.5, att_map_uint8, 0.5, 0.0)
            augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
            augmented = np.moveaxis(augmented, (0, 1, 2), (1, 2, 0)) / 255.
            augmented_images.append(torch.tensor(augmented))

    return augmented_images


def get_mask(coco, ann_id):
    ann = coco.anns[ann_id]
    img_id = ann['image_id']
    category_id = ann['category_id']
    print('img_id: {}\ncategory_id: {}'.format(img_id, category_id))
    m = coco.annToMask({'image_id': img_id, 'segmentation': coco.annToRLE(coco.anns[ann_id])})
    return m


def get_color_segmentation(img, markers=128):
    from skimage.filters import sobel
    from skimage.color import rgb2gray
    from skimage.segmentation import watershed

    gradient = sobel(rgb2gray(img))
    segments = watershed(gradient, markers=markers, compactness=0.001)
    segments -= 1
    binaries = np.zeros([np.max(segments), img.shape[0], img.shape[1]], dtype=np.bool)
    segment_sizes = np.zeros([np.max(segments)], dtype=np.float32)
    num_pixels = list()
    for i in range(np.max(segments)):
        binary = segments == i
        segment_sizes[i] = binary.sum()
        binaries[i] = binary
    segments = np.stack(binaries)
    return segments, segment_sizes


def get_selective_search_proposals(img):
    import selectivesearch
    # img = cv2.rectangle(img, region['rect'], (255, 0, 0), 1)
    _, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=25)
    r_idxs = np.random.permutation(len(regions))

    while len(r_idxs) < 64:
        r_idxs = np.concatenate([r_idxs, np.random.permutation(len(regions))])

    num_proposals = 64
    regions = [regions[r_idxs[r]] for r in range(np.minimum(len(r_idxs), num_proposals))]
    proposals = np.zeros([num_proposals, img.shape[0], img.shape[1]], dtype=np.bool)
    proposal_sizes = np.zeros([num_proposals], dtype=np.float32)
    boxes = list()
    for r_idx, region in enumerate(regions):
        x, y, h, w = region['rect']
        boxes.append([x, y, x + h, y + w])
        h = h + 1
        w = w + 1
        proposal_sizes[r_idx] = h * w
        proposals[r_idx, x:x + h, y:y + w] = True

    # plot_image(proposals.sum(dim=(0, 1)))

    return proposals, proposal_sizes, boxes


def get_color_segmentations(batch_imgs, markers=128):

    seg_maps = np.zeros((batch_imgs.shape[0], batch_imgs.shape[2], batch_imgs.shape[3]))
    batch_imgs = batch_imgs.numpy()
    for img_idx in range(batch_imgs.shape[0]):
        img = batch_imgs[img_idx]
        img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
        seg_maps[img_idx] = get_color_segmentation(img, markers)

    return torch.tensor(seg_maps)


def rand_targets_to_label(rand_target, data_loader):
    class_idx = rand_target.max(dim=0)[1].item()
    class_label = data_loader.dataset.cats_dict[class_idx]
    print('{} {}'.format(class_idx, class_label))
    return class_label


def plotter(idx, sub_idx, image_list, name_list,
            rand_targets, data_loader):
    class_label = rand_targets_to_label(rand_targets[idx], data_loader)

    for i, (name, images) in enumerate(zip(name_list, image_list)):
        if images.shape[1] == 1 or images.shape[1] == 3 or len(images.shape) == 3:
            plot(images[idx], i, title='{} {}'.format(name, class_label))
        else:
            plot(images[idx, sub_idx], i, title='{} {}'.format(name, class_label))

# utils.plotter(2, 55, [source_images, color_seg_maps, maps, seg_maps, brushed_class_maps, brushed_seg_logits],
# ['source_images', 'color_seg_maps', 'maps', 'seg_maps', 'brushed_class_maps', 'brushed_seg_logits'], rand_targets, data_loader)


def brush_map(color_seg_maps, att_orig):
    original_shape = att_orig.shape[2:]
    # att_orig = F.interpolate(att_orig, size=(224, 224), mode='bilinear', align_corners=False)
    final_map = torch.zeros(att_orig.shape, device=device)
    for (seg_idx, m) in enumerate(color_seg_maps):
        for i in range(1, int(torch.max(m).cpu().numpy())):
            binary = m == i
            num_pixels = torch.sum(binary)
            binary_mask = binary.view(-1, binary.shape[0], binary.shape[1]).type(torch.float)
            selection = binary_mask * att_orig[seg_idx]
            avg_value = selection.sum(dim=(1, 2), keepdim=True) / num_pixels
            final_map[seg_idx] = final_map[seg_idx] + binary_mask * avg_value
    # final_map = F.interpolate(final_map, size=original_shape, mode='bilinear', align_corners=False)
    return final_map  # F.softmax(final_map, dim=1)


def normalize_image(images):
    if len(images.shape) >= 4:
        images_shape = images.shape
        work_images = images.view(images_shape[0], images_shape[1], -1)
        max_value = work_images.max(dim=2, keepdims=True)[0]
        min_value = work_images.min(dim=2, keepdims=True)[0]
        work_images = (work_images - min_value)/(max_value - min_value)
        work_images = work_images.view(images_shape)
    else:
        raise ValueError('Not implemented.')

    return work_images


# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def bb_intersection_over_AorB(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def brush_map2(color_seg_maps, att1, att2=None):
    final_map = torch.zeros(att1.shape, device=device)
    if att2 is not None:
        final_map2 = torch.zeros(att1.shape, device=device)
    for i in range(1, int(torch.max(color_seg_maps).item()) + 1):
        binary = color_seg_maps == i
        num_pixels = torch.sum(binary, dim=(1, 2)).type(torch.float)
        binary_mask = binary.view(-1, 1, binary.shape[1], binary.shape[2]).type(torch.float)

        selection = binary_mask * att1
        avg_value = selection.sum(dim=(2, 3), keepdim=True) / num_pixels.view(-1, 1, 1, 1)
        final_map = final_map + binary_mask * avg_value

        if att2 is not None:
            selection2 = binary_mask * att2
            avg_value2 = selection2.sum(dim=(2, 3), keepdim=True) / num_pixels.view(-1, 1, 1, 1)
            final_map2 = final_map2 + binary_mask * avg_value2

    if att2 is not None:
        return final_map, final_map2
    else:
        return final_map


def superpixel_map(color_seg_maps, att1):
    values = []
    # boxs = []
    # final_map = torch.zeros(att1.shape, device=device)
    for i in range(1, int(torch.max(color_seg_maps).item()) + 1):
        binary = color_seg_maps == i
        num_pixels = torch.sum(binary, dim=(1, 2)).type(torch.float)
        binary_mask = binary.view(-1, 1, binary.shape[1], binary.shape[2]).type(torch.float)

        # h_tol, w_tol = binary_mask.shape[2:]

        # torch_zero = torch.zeros((1), device=device)
        # h = torch.ne(binary_mask.sum(dim=2, keepdim=True), torch_zero).type(torch.float)
        # c = torch.ones(size=h.shape, device=device).cumsum(dim=3)
        # hc = torch.where(h != 0, h + c.flip(dims=[3]), h)
        # h_min = hc.max(dim=3)[1].squeeze()
        # hc = torch.where(h != 0, h+c, h)
        # h_max = hc.max(dim=3)[1].squeeze()
        #
        # w = torch.ne(binary_mask.sum(dim=3, keepdim=True), torch_zero).type(torch.float)
        # c = c.permute(dims=(0, 1, 3, 2))
        # wc = torch.where(w != 0, w + c.flip(dims=[2]), w)
        # w_min = wc.max(dim=2)[1].squeeze()
        # wc = torch.where(w != 0, w + c, w)
        # w_max = wc.max(dim=2)[1].squeeze()

        # h_min = h_min/float(h_tol)
        # h_max = h_max / float(h_tol)
        # w_min = w_min / float(w_tol)
        # w_max = w_max / float(w_tol)
        #
        # hw_labels = torch.stack([h_min, w_min, h_max - h_min, w_max - w_min], dim=1)

        # box = torch.stack([h_min, w_min, h_max, w_max], dim=1)
        # boxs.append(box)

        # img = binary_mask[1].detach().cpu().numpy() * 255
        # img = np.concatenate([img, img, img], axis=0)
        # img = np.swapaxes(img, 0, 1).swapaxes(1, 2)
        # img = img.astype(np.uint8)
        # import cv2
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # some_img = np.zeros(img.shape, dtype=np.uint8)
        # some_img = cv2.rectangle(some_img, (h_min[1].item(), w_min[1].item()), (h_max[1].item(), w_max[1].item()), (255, 0, 0), 1)
        # img = cv2.addWeighted(some_img, 0.5, img, 0.5, 0.0)
        # utils.plot_image(img)
        # utils.plot_image(binary_mask.sum(dim=2, keepdim=True).repeat(1, 1, 224, 1)[1], 2)

        selection = binary_mask * att1
        avg_value = selection.sum(dim=(2, 3), keepdim=True) / num_pixels.view(-1, 1, 1, 1)
        # final_map = final_map + binary_mask * avg_value
        values.append(avg_value.squeeze())
    values = torch.stack(values, dim=1)
    det_logits = torch.max(values, dim=1)[0]
    cls_probs = torch.softmax(values, dim=2)
    cls_probs = cls_probs.mean(dim=1)
    # boxs = torch.stack(boxs, dim=2).permute([0, 2, 1])
    return det_logits, cls_probs


# Copied from: https://github.com/utkuozbulak/pytorch-cnn-visualizations
# https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/misc_functions.py
def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def grayscale_grad(rgb_grad):
    if len(rgb_grad.shape) != 4:
        raise ValueError('The input must be a 4-D tensor')

    if rgb_grad.shape[1] != 3:
        raise ValueError('The input tensor should be in batch_size x (channel=3) x h x w format')

    gs_grad = torch.abs(rgb_grad).sum(dim=1, keepdim=True)
    return gs_grad


matlab_colors = [
   [0, 114, 189],
   [217, 83, 25],
   [237, 177, 32],
   [126, 47, 142],
   [119, 172, 48],
   [77, 190, 238],
   [162, 20, 47]
]