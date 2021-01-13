# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import sys
from projector import *

def main():

    seed = 303
    minibatch_size = 10
    network_pkl = 'training-runs/00002-djokovic_fg_black_v2_tf-mirror-paper256/network-snapshot-022937.pkl'
    images_dir = 'datasets/djokovic_fg_black_v2'
    output_dir = 'result/embed_djokovic_black_v2_22937'
    os.makedirs(output_dir, exist_ok=True)

    split_idx = int(sys.argv[1])
    num_split = int(sys.argv[2])

    # Load networks.
    tflib.init_tf({'rnd.np_random_seed': seed})
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    # Initialize projector.
    proj = Projector()
    proj.set_network(Gs, minibatch_size)
    proj.num_steps = 1000

    images_name = sorted(os.listdir(images_dir))

    result_dict = {}
    for batch_idx in range(split_idx * minibatch_size, len(images_name), num_split * minibatch_size):
        print("Split {} running batch {} of {}".format(split_idx, batch_idx, len(images_name) // minibatch_size // num_split))

        images_name_batch = images_name[batch_idx: batch_idx + minibatch_size]
        images_gt = [np.array(PIL.Image.open(os.path.join(images_dir, f))) for f in images_name_batch]
        images_float = [image.astype(np.float32).transpose([2, 0, 1]) * (2 / 255) - 1 for image in images_gt] 

        proj.start(images_float)
        # Run projector.
        with tqdm.trange(proj.num_steps) as t:
            for step in t:
                assert step == proj.cur_step
                dist, loss = proj.step()
                t.set_postfix(dist=f'{dist[0]:.4f}', loss=f'{loss:.2f}')

        # Save results.
        # imgs_embed = proj.images_uint8
        for i, f in enumerate(images_name_batch):
            result_dict[f] = proj.dlatents[i]

        pickle.dump(result_dict, open(os.path.join(output_dir, 'dlatents-{}-{}.pkl'.format(split_idx, num_split)), 'wb'))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
