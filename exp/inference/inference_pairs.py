import socket
import pickle
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
from collections import OrderedDict

sys.path.append("./")
# PyTorch includes
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import cv2


# Custom includes
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr
from guided_filter import GuidedFilter
#
import argparse
import torch.nn.functional as F




label_colours = [
    (0, 0, 0),
    (128, 0, 0),
    (255, 0, 0),
    (0, 85, 0),
    (170, 0, 51),
    (255, 85, 0),
    (0, 0, 85),
    (0, 119, 221),
    (85, 85, 0),
    (0, 85, 85),
    (85, 51, 0),
    (52, 86, 128),
    (0, 128, 0),
    (0, 0, 255),
    (51, 170, 221),
    (0, 255, 255),
    (85, 255, 170),
    (170, 255, 85),
    (255, 255, 0),
    (255, 170, 0),
]


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]


def flip_cihp(tail_list):
    """

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    """
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
    return torch.cat(tail_list_rev, dim=0)


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
    assert n >= num_images, (
        "Batch size %d should be greater or equal than number of images to save %d."
        % (n, num_images)
    )
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new("RGB", (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


def read_img(img_path):
    _img = Image.open(img_path).convert("RGB")  # return is RGB pic
    return _img


def img_transform(img, transform=None):
    sample = {"image": img, "label": 0}

    sample = transform(sample)
    return sample


def inference(net, img_path="", output_path="./", output_name="f", use_gpu=True):
    """

    :param net:
    :param img_path:
    :param output_path:
    :return:
    """

    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = (
        adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)
    )

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    # multi-scale
    scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    real_img = read_img(img_path)

    img = real_img.resize((256, 256))
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose(
            [
                tr.Scale_only_img(pv),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img(),
            ]
        )

        composed_transforms_ts_flip = transforms.Compose(
            [
                tr.Scale_only_img(pv),
                tr.HorizontalFlip_only_img(),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img(),
            ]
        )

        testloader_list.append(img_transform(img, composed_transforms_ts))
        # print(img_transform(img, composed_transforms_ts))
        testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
    # print(testloader_list)
    start_time = timeit.default_timer()
    # One testing epoch
    net.eval()
    # 1 0.5 0.75 1.25 1.5 1.75 ; flip:

    for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
        inputs, labels = sample_batched[0]["image"], sample_batched[0]["label"]
        inputs_f, _ = sample_batched[1]["image"], sample_batched[1]["label"]
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
            outputs = net.forward(
                inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda()
            )
            outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
            outputs = outputs.unsqueeze(0)

            if iii > 0:
                outputs = F.upsample(
                    outputs, size=(h, w), mode="bilinear", align_corners=True
                )
                outputs_final = outputs_final + outputs
            else:
                outputs_final = outputs.clone()

    #foreground_unroll = torch.resize(foreground_unroll, (H, W))
    
    ################ plot pic
    predictions = torch.max(outputs_final, 1)[1]
    results = predictions.cpu().numpy()
    vis_res = decode_labels(results)

    parsing_im = Image.fromarray(vis_res[0])
    parsing_im.save(os.path.join(output_path, output_name + '.png'))
    #cv2.imwrite("outputs/{}_gray.png".format(output_name), results[0, :, :])

    end_time = timeit.default_timer()
    print(
        "time used for the multi-scale image inference"
        + " is :"
        + str(end_time - start_time)
    )


    # guided filter code goes here
    guidance_map = real_img.resize((512, 512))
    guidance_map = transforms.ToTensor()(guidance_map)
    guidance_map = guidance_map.unsqueeze(0).cuda()
    
    guide = GuidedFilter(
        #guidance_map.shape[-2:],
        r=4,
        eps=1e-3,
        downsample_stride=4,
    ).cuda()
    guide.conv_mean_weights = guide.conv_mean_weights.cuda()

    predictions = F.interpolate(
        outputs_final,
        size=guidance_map.shape[-2:],
        mode='bilinear',
        align_corners=False
    )

    # get guided predictions; 
    # guidance_map: [1, 3, 512, 512], predictions: [1, 20, 512, 512]
    for idx, channel in enumerate(predictions[0, ...]):
        current_channel = predictions[:, idx, :, :]
        current_channel = torch.unsqueeze(current_channel, axis=0) # [1, 1, 512, 512]
        if idx == 0:
            guided_predictions = guide(guidance_map, current_channel) # [1, 1, 512, 512]
        else:
            guided_predictions = torch.cat(
                (guided_predictions, guide(guidance_map, current_channel)),
                axis=1
            )
    # guided_predictions: [1, 20, 512, 512]
    guided_predictions = F.softmax(guided_predictions, dim=1)
    foreground_predictions = guided_predictions[0, 1:, ...] # [19, 512, 512], 0 is background
    foreground_soft_mask = torch.sum(foreground_predictions, axis=0) # [H, W]
    foreground_soft_mask = torch.unsqueeze(foreground_soft_mask, axis=-1) # [H, W, 1]
    foreground_soft_mask = torch.cat(
        (foreground_soft_mask, foreground_soft_mask, foreground_soft_mask),
        axis=-1
    )
    foreground_soft_mask = foreground_soft_mask.cpu().numpy()
    with open(os.path.join(output_path, 'FG_softmask_' + output_name + '.pkl'), 'wb') as handle:
        pickle.dump(foreground_soft_mask, handle)
    foreground_soft_mask = (foreground_soft_mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, 'guided_FG_' + output_name + '.png'), foreground_soft_mask)

    hair_predictions = guided_predictions[0, 2, :, :]
    face_predictions = guided_predictions[0, 13, :, :]
    foreground_softmask = hair_predictions + face_predictions # [512, 512]
    foreground_softmask = torch.unsqueeze(foreground_softmask, axis=-1)
    foreground_softmask = torch.cat(
        (foreground_softmask, foreground_softmask, foreground_softmask),
        axis=-1
    )
    foreground_softmask = foreground_softmask.cpu().numpy()
    with open(os.path.join(output_path, 'HF_softmask_' + output_name + '.pkl'), 'wb') as handle:
        pickle.dump(foreground_softmask, handle)
    foreground_softmask = (foreground_softmask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, 'guided_HF_' + output_name + '.png'), foreground_softmask)


if __name__ == "__main__":
    """argparse begin"""
    parser = argparse.ArgumentParser()
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument("--loadmodel", default="", type=str)
    parser.add_argument("--img_path", default="", type=str)
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--output_name", default="", type=str)
    parser.add_argument("--use_gpu", default=1, type=int)
    opts = parser.parse_args()

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
        n_classes=20, hidden_layers=128, source_classes=7,
    )
    if not opts.loadmodel == "":
        x = torch.load(opts.loadmodel)
        net.load_source_model(x)
        print("load model:", opts.loadmodel)
    else:
        print("no model load !!!!!!!!")
        raise RuntimeError("No model!!!!")

    if opts.use_gpu > 0:
        net.cuda()
        use_gpu = True
    else:
        use_gpu = False
        raise RuntimeError("must use the gpu!!!!")

    path = opts.img_path
    I_1_path = os.path.join(path, 'I_1.png')
    synth_path = os.path.join(path, 'project.png')

    inference(
        net=net,
        img_path=I_1_path,
        output_path=path,
        output_name='I_1_M',
        use_gpu=use_gpu,
    )

    inference(
        net=net,
        img_path=synth_path,
        output_path=path,
        output_name='project_M',
        use_gpu=use_gpu,
    )
