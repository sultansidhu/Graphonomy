import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
from collections import OrderedDict
sys.path.append('./')
# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2


# Custom includes
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

#
import argparse
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#label_colours = [(0,0,0)
#                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
#                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

# Setting all labels to 0, except hair and face
label_colours = [(0, 0, 0) for i in range(20)]
label_colours[2] = (255, 0, 0)
label_colours[13] = (0, 0, 255)

def overlay(frame, mask):

    mask = np.array(mask)
    frame = np.array(frame)

    tmp = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(mask)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)

    # overlay mask on frame
    overlaid_image = cv2.addWeighted(frame,0.4,dst,0.1,0)
    return overlaid_image

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def read_img(img_path):
    _img = Image.open(img_path).convert('RGB')  # return is RGB pic
    return _img

def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample

def inference(net, img=None, use_gpu=True):
    '''

    :param net:
    :return:
    '''
    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    # multi-scale
    scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    # img = read_img(img_path)
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        composed_transforms_ts_flip = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.HorizontalFlip_only_img(),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        testloader_list.append(img_transform(img, composed_transforms_ts))
        # print(img_transform(img, composed_transforms_ts))
        testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
    # print(testloader_list)
    start_time = timeit.default_timer()
    # One testing epoch
    net.eval()
    # 1 0.5 0.75 1.25 1.5 1.75 ; flip:

    for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
        inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
        inputs = inputs.unsqueeze(0)
        inputs_f = inputs_f.unsqueeze(0)
        inputs = torch.cat((inputs, inputs_f), dim=0)
        if iii == 0:
            _, _, h, w = inputs.size()
        # assert inputs.size() == inputs_f.size()

        # Forward pass of the mini-batch
        inputs = Variable(inputs, requires_grad=False)

        with torch.no_grad():
            if use_gpu >= 0:
                inputs = inputs.cuda()
            # outputs = net.forward(inputs)
            outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
            outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
            outputs = outputs.unsqueeze(0)

            if iii > 0:
                outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                outputs_final = outputs_final + outputs
            else:
                outputs_final = outputs.clone()
    ################ plot pic
    predictions = torch.max(outputs_final, 1)[1]
    results = predictions.cpu().numpy()
    vis_res = decode_labels(results)

    parsing_im = Image.fromarray(vis_res[0])
    return parsing_im
    # parsing_im.save(output_path+'/{}.png'.format(output_name))
    # cv2.imwrite(output_path+'/{}_gray.png'.format(output_name), results[0, :, :])

    # end_time = timeit.default_timer()
    # print('time used for the multi-scale image inference' + ' is :' + str(end_time - start_time))

if __name__ == '__main__':
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument('--loadmodel', default='', type=str)
    parser.add_argument('--use_gpu', default=1, type=int)
    parser.add_argument(
        '--BASE_DIR',
        default='/mnt/Data/Data/modidatasets/VoxCeleb2/',
        type=str)
    opts = parser.parse_args()

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                 hidden_layers=128,
                                                                                 source_classes=7, )
    if not opts.loadmodel == '':
        x = torch.load(opts.loadmodel)
        net.load_source_model(x)
        print('load model:', opts.loadmodel)
    else:
        print('no model load !!!!!!!!')
        raise RuntimeError('No model!!!!')

    if opts.use_gpu >0 :
        net.cuda()
        use_gpu = True
    else:
        use_gpu = False
        raise RuntimeError('must use the gpu!!!!')


    # reading frames from videos
    BASE_DIR = opts.BASE_DIR 
    TRAIN_DIR = os.path.join(BASE_DIR, 'train')
    ids = sorted(os.listdir(TRAIN_DIR))

    for i in ids[8:20]:
        id_path = os.path.join(TRAIN_DIR, i)

        codes = os.listdir(id_path)

        for code in codes:
            code_path = os.path.join(id_path, code)

            mp4s = os.listdir(code_path)
            print('Processing:{}, {}'.format(i, code))

            for mp4 in mp4s:
                mp4_path = os.path.join(code_path, mp4)

                # processed image and mask directory path
                SAVE_DIR = os.path.join(
                    BASE_DIR, 'processed_train', i, code,
                    mp4.split('.')[0])

                if not os.path.exists(SAVE_DIR):
                    os.makedirs(SAVE_DIR)

                # read frames using cv2
                cap = cv2.VideoCapture(mp4_path)
                if cap.isOpened() == False:
                    print('Error reading {}'.format(mp4_path))
                else:
                    img_ctr = 1
                    while cap.isOpened():
                        ret, frame = cap.read()

                        if ret == True:
                            frame = cv2.cvtColor(
                                frame,
                                cv2.COLOR_BGR2RGB)
                            frame = Image.fromarray(frame)

                            mask = inference(
                                net=net,
                                img=frame,
                                use_gpu=use_gpu)
    
                            mask_filename = '{}_mask.png'.format(str(img_ctr))
                            frame_filename = '{}_frame.png'.format(str(img_ctr))
                            '''
                            if img_ctr == 1:
                                # cv2 video writer init
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                width, height = frame.size
                                video_writer = cv2.VideoWriter(
                                    os.path.join(MASK_DIR, 'mask_overlaid_video.mp4'),
                                    fourcc, 25, (width, height))

                            mask_alpha = mask.convert('RGBA')
                            frame_alpha = frame.convert('RGBA')
                            overlaid_image = Image.blend(frame, mask, 0.5)

                            video_writer.write(np.array(overlaid_image))
                            '''

                            frame.save(
                                os.path.join(SAVE_DIR, frame_filename))
                            mask.save(
                                os.path.join(SAVE_DIR, mask_filename))

                            img_ctr += 1

                        else:
                            break
                            
                    cap.release()
                    # video_writer.release()
