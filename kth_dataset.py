import itertools

from FloorPlanGraph import *

from data_generators.augmentable_dataset import AugmentableDataset

class KTHFloorplanDataset(AugmentableDataset):
    def __init__(self, dataset_dir: typing.Type[str],
                 map_resolution: typing.Type[float]=0.05,
                 sampling_distance: typing.Type[float]=1,
                 complete_map_size: typing.Tuple[int, int]=(128, 128),
                 partial_map_size: typing.Tuple[int, int]=(64, 64)):
        """
        dataset constructor
        :param dataset_dir: directory for reading xml files of kth dataset
        :param map_resolution: m/pixel of the map
        :param sampling_distance: sampling rate for generating maps
        :param complete_map_size: size of the complete map (input to network)
        :param partial_map_size: size of the partial map (the portion of the input map which contains actual values)
        """
        self.dataset_dir = dataset_dir
        self.map_resolution = map_resolution
        self.sampling_distance = sampling_distance
        self.complete_map_size = complete_map_size
        self.partial_map_size = partial_map_size

        self.floorplan_dataset: [FloorPlanGraph] = KTHFloorplanDataset._readAllfileInfold(sdir=self.dataset_dir,
                                                                                          rootNodeName="floor")
        self.floorplan_sample_lens: [int] = KTHFloorplanDataset._get_sample_lens(self.floorplan_dataset,
                                                                                 self.sampling_distance)

        self.accumulate_lens = itertools.accumulate(self.floorplan_sample_lens)
        self.samples_total_lens = self.accumulate_lens[-1]  # last element of accumulate_lens is total numbers of samples

    @classmethod
    def _readAllfileInfold(cls, sdir: str, rootNodeName: str) -> [FloorPlanGraph]:
        floorplans = []

        if not os.path.exists(sdir) or not os.path.isdir(sdir):
            return

        for sub in [os.path.join(sdir, o) for o in os.listdir(sdir)]:
            if os.path.isdir(sub):
                floorplans_tmp = cls._readAllfileInfold(sdir=sub,rootNodeName=rootNodeName)
                floorplans.extend(floorplans_tmp)
            elif os.path.isfile(sub) and os.path.splitext(sub)[1] == ".xml":
                # check if it is a file and its extension name is .xml
                floorplan = FloorPlanGraph()
                floorplan.loadFromXml(filenamePath=sub, rootNodeName=rootNodeName)
                floorplans.append(floorplan)

        return floorplans

    @classmethod
    def _get_sample_lens(cls, floorplans: [FloorPlanGraph], sampling_distance):
        floorplan_lens = len(floorplans)
        sample_lens = []
        for index in range(0, floorplan_lens):
            floorplan: FloorPlanGraph = floorplans[i]
            lens = floorplan.size_samples(sample_distance=sampling_distance)
            sample_lens.append(lens)

        return sample_lens

    def __len__(self):
        return self.samples_total_lens

    def __index_of_sample(self, index) -> (int, int):
        """
        get the index of floorplan and the index of sample in this floorplan
        :param index: index in all samples
        :return: (int, int) -> (floorplan_index, sample_index_in_floorplan)
        """
        if index >= self.samples_total_lens or index < 0:
            raise IndexError("Error index when get item.")

        floorplan_index = 0
        sample_index = 0
        while not index < self.accumulate_lens[floorplan_index]:
            floorplan_index += 1

        if floorplan_index != 0:
            sample_index = index - self.accumulate_lens[floorplan_index - 1]
        else:
            sample_index = index

        return floorplan_index, sample_index

    def __getitem__(self, index):
        """
        return a item of KTH_floor_plan_dataset, which is a item
        :param index: index of the item
        :return: 2-tuple (incomplete map, complete map)
        """
        floorplan_index, sample_index = self.__index_of_sample(index)
        complete_map = self.floorplan_dataset[floorplan_index].get_sample(sample_index,
                                                                          target_resolution=self.map_resolution,
                                                                          image_size=self.complete_map_size,
                                                                          sample_distance=self.sampling_distance)
        # center crop
        start = (np.asarray(self.complete_map_size) / 2.0 - np.asarray(self.partial_map_size) / 2.0).astype(np.int)
        end = (np.asarray(self.complete_map_size) / 2.0 + np.asarray(self.partial_map_size) / 2.0).astype(np.int)
        padded_partial_map = np.ones(self.complete_map_size, dtype=np.float32) * 0.5
        padded_partial_map[start[0]:end[0], start[1]:end[1]] = complete_map[start[0]:end[0], start[1]:end[1]]

        return padded_partial_map, complete_map

if __name__== "__main__":
    import sys
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description='KTH datagenerator test')
    parser.add_argument(
        '--sampling-distance', type=float, default=25.0,
        metavar='N', help='sampling distance for generating data, default 25.0'

    )
    parser.add_argument(
        '--map-resolution', type=float, default=0.2,
        metavar='N', help='resolution of the map to output (m/pixel), default 0.2'
    )

    parser.add_argument(
        '--shuffle', dest='is_shuffle', action='store_true',
        help='shuffle the data (default true)'
    )
    parser.add_argument(
        '--no-shuffle', dest='is_shuffle', action='store_false',
        help='do not shuffle the data (default true)'
    )
    parser.set_defaults(is_shuffle=True)

    parser.add_argument(
        '--batch-size', type=int, default=16,
        metavar='N', help='batch size'
    )
    parser.add_argument(
        'dataset_dir', type=str, default='.',
        metavar='S', help='directory of the dataset'
    )
    args = parser.parse_args()

    annotation_folder = sys.argv[1]
    kth_floorplan = KTHFloorplanDataset(args.dataset_dir, sampling_distance=args.sampling_distance, map_resolution=args.map_resolution)

    dataloader = DataLoader(kth_floorplan, batch_size=args.batch_size, shuffle=args.is_shuffle)

    for batch_idx, data in enumerate(dataloader):
        mixed_data = torch.FloatTensor(2 * args.batch_size, data[0].size(-2), data[0].size(-1))
        mixed_data[0::2, :, :] = data[0]
        mixed_data[1::2, :, :] = data[1]

        grid = torchvision.utils.make_grid(mixed_data.unsqueeze(1), nrow=int(args.batch_size / 2), padding=10)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        plt.imshow(ndarr)
        plt.show()