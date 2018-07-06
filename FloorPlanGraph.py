from GraphFileOperations import *
from enum import Enum
import cv2
import typing
import numpy as np


class Point2D(object):
    def __init__(self, x: float = 0.0, y: float = 0.0) -> object:
        self.x = x  # type: float
        self.y = y  # type: float

    def __add__(self, p):  # p is Point2D
        return Point2D(self.x + p.x, self.y + p.y)

    def __sub__(self, p):  # p is Point2D
        return Point2D(self.x - p.x, self.y - p.y)

    def __truediv__(self, op: float):
        return Point2D(self.x / op, self.y / op)

    def __mul__(self, op: float):
        return Point2D(self.x * op, self.y * op)


class LineSegment(object):
    def __init__(self, startPos = Point2D(), endPos = Point2D()):
        self.startPos = startPos
        self.endPos = endPos
        self.type = ""
        self.portalToRoom = ""


class Space(object):
    def __init__(self):
        self.category = ""
        self.vertex_id = ""
        self.roomLayout: [LineSegment] = []  # type: [LineSegment]
        self.maxx = 0.0
        self.maxy = 0.0
        self.minx = 0.0
        self.miny = 0.0
        self.centroid = Point2D()

    def updateExtent(self):
        self.maxx = float("-inf")
        self.maxy = float("-inf")
        self.minx = float("inf")
        self.miny = float("inf")

        for line_seg in self.roomLayout:
            cur_minx = min(line_seg.startPos.x, line_seg.endPos.x)
            cur_miny = min(line_seg.startPos.y, line_seg.endPos.y)
            cur_maxx = max(line_seg.startPos.x, line_seg.endPos.x)
            cur_maxy = max(line_seg.startPos.y, line_seg.endPos.y)

            if cur_minx < self.minx:
                self.minx = cur_minx

            if cur_miny < self.miny:
                self.miny = cur_miny

            if cur_maxx > self.maxx:
                self.maxx = cur_maxx

            if cur_maxy > self.maxy:
                self.maxy = cur_maxy


class EdgeClass(Enum):
    HORIZONTAL = 1
    VERTICAL = 2


class EdgeType(Enum):
    EXPLICIT_EDGE = 1
    IMPLICIT_EDGE = 2


class spaceEdge(object):
    def __init__(self):
        self.edge_id = ""
        self.edge_class = EdgeClass.HORIZONTAL
        self.edge_type = EdgeType.EXPLICIT_EDGE

class graphProperties(object):
    def __init__(self):
        self.floorname = ""
        self.filepath = ""
        self.maxx = 0.0
        self.maxy = 0.0
        self.minx = 0.0
        self.miny = 0.0
        self.centroid = Point2D()
        self.pixel_distance = 0.0
        self.real_distance = 0.0

class FloorPlanGraph(object):
    def __init__(self):
        self.graph = None

    def loadFromXml(self, filenamePath: str, rootNodeName: str= "floor"):
        graph, graph_property = GraphFileOperations.loadGraphFromXML(filenamePath, rootNodeName)
        self.graph = graph
        self.updateExtent()

    def updateExtent(self):
        maxx = float("-inf")
        maxy = float("-inf")
        minx = float("inf")
        miny = float("inf")

        for spaceName in self.graph.nodes():
            space = self.graph.nodes[spaceName]["attr"]
            minx = min(minx, space.minx)
            miny = min(miny, space.miny)
            maxx = max(maxx, space.maxx)
            maxy = max(maxy, space.maxy)

        self.graph.graph["attr"].minx = minx
        self.graph.graph["attr"].miny = miny
        self.graph.graph["attr"].maxx = maxx
        self.graph.graph["attr"].maxy = maxy

        self.graph.graph["attr"].centroid.x = (minx + maxx)/2
        self.graph.graph["attr"].centroid.y = (miny + maxy)/2

    def get_real_size(self) -> (float, float):
        """
        get the real x-y distance (m) of the floorplan
        :return: x-length (width) and y-length (height) of the floorplan
        """
        graph_property = self.graph.graph["attr"]  # type: graphProperties
        resolution = self.get_resolution()
        xl = (graph_property.maxx - graph_property.minx) * resolution
        yl = (graph_property.maxy - graph_property.miny) * resolution
        return xl, yl

    def get_resolution(self):
        """
        get the resolution (m/pixel) of the floorplan
        :return: resolution of the floorplan (m / pixel)
        """
        graph_property = self.graph.graph["attr"]
        resolution = graph_property.real_distance / graph_property.pixel_distance
        return resolution

    def get_map(self, sample_centroid: (float, float),
                target_resolution: float, image_size: (float, float)):
        """
        return a map centered at sample_centroid
        :param sample_centroid: the center (pixels coordinate) of sampled (partial) area, using original resolution
        :param target_resolution: the resolution of output image, resolution: m / pixels
        :param image_size: height and width of the output image
        :return: a (partial) image to represent the map
        """
        out_image = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        original_resolution = self.get_resolution()

        sample_centroid = Point2D(sample_centroid[0], sample_centroid[1])
        map_centroid = Point2D(image_size[0] / 2, image_size[1] / 2)

        def draw_segment(seg: LineSegment, color):
            start = (seg.startPos - sample_centroid) * original_resolution / target_resolution + map_centroid
            end = (seg.endPos - sample_centroid) * original_resolution / target_resolution + map_centroid

            start_int = (round(start.x), round(start.y))
            end_int = (round(end.x), round(end.y))

            cv2.line(out_image, start_int, end_int, color=color)

        for spaceName in self.graph.nodes():
            space: Space = self.graph.nodes[spaceName]["attr"]
            doors = []
            for segment in space.roomLayout:
                if segment.type == "Portal":
                    doors.append(segment)
                    continue
                else:
                    draw_segment(segment, color=1.0)

            for segment in doors:
                draw_segment(segment, color=0)

        return out_image


    def to_image(self, target_resolution=0.2, image_size: typing.Tuple[int, int]=(1000, 1000)):
        """
        output the whole image of the map, if the resolution is too precise that cannot hold the full map
        within map_size, the map will be cropped
        :param image_size: the height and width of the map
        :param target_resolution: resolution of the map
        :return: image to represent the map
        """
        graph_property = self.graph.graph["attr"]  # type: graphProperties
        centroid = graph_property.centroid
        return self.get_map((centroid.x, centroid.y), target_resolution, image_size)

    def get_samples(self, target_resolution: float=0.5, image_size: (float, float)=(256, 256),
                            sample_distance=0.5):
        """
        sample partial maps with target_resolution and image_size in uniform way
        :param target_resolution: output image resolution (m/pixel)
        :param image_size: output image size
        :param sample_distance: distances between two sample map centroids (m)
        :return: all sample images
        """
        sample_size = self.size_samples(sample_distance=sample_distance)
        resolution = self.get_resolution()

        graph_property = self.graph.graph["attr"]  # type: graphProperties
        map_min = (graph_property.minx, graph_property.miny)

        samples = []
        pixel_sample_distance = sample_distance/resolution

        for i in range(0, sample_size[0]):
            for j in range(0, sample_size[1]):
                sample_centroid = (int(map_min[0] + i*pixel_sample_distance),
                                   int(map_min[1] + j*pixel_sample_distance))
                single_sample = self.get_map(sample_centroid=sample_centroid, image_size=image_size,
                                             target_resolution=target_resolution)
                samples.append(single_sample)

        return samples

    def size_samples(self, sample_distance):
        """
        get size (x*y) of samples with sample_distance
        :param sample_distance: distance interval to sample a points
        :return: sample_size
        """
        map_real_size = self.get_real_size()
        sample_size = (int(map_real_size[0]/sample_distance), int(map_real_size[1]/sample_distance))
        return sample_size

    def len_samples(self, sample_distance):
        """
        get numbers of samples with sample_distance
        :param sample_distance: distance to sample a points
        :return: numbers of samples
        """
        sample_size = self.size_samples(sample_distance)
        return sample_size[0] * sample_size[1]

    def get_sample(self, index: int, target_resolution: float=0.5, image_size: (float, float)=(256, 256),
                   sample_distance=0.5):
        """
        Get a sample with index
        :param index: the index of the sample in all samples
        :param target_resolution: output image resolution (m/pixel)
        :param image_size: output image size
        :param sample_distance: distances between two sample map centroids (m)
        :return: a sample
        """
        sample_size = self.size_samples(sample_distance)
        x_index = index // sample_size[1]
        y_index = index % sample_size[1]

        if x_index >= sample_size[0]:
            raise IndexError("sample index is out of range")

        graph_property = self.graph.graph["attr"]  # type: graphProperties
        map_min = (graph_property.minx, graph_property.miny)

        pixel_sample_distance = sample_distance / self.get_resolution()
        sample_centroid = (int(map_min[0] + x_index * pixel_sample_distance),
                           int(map_min[1] + y_index * pixel_sample_distance))
        single_sample = self.get_map(sample_centroid=sample_centroid, image_size=image_size,
                                     target_resolution=target_resolution)
        return single_sample


if __name__ == '__main__':
    testx = Point2D()

    print(testx.x)
    print(testx.y)

    testx.x = 1
    testx.y = 2

    y = Point2D(3, 4)
    print(y.x)
    print(y.y)

    z = testx * 3
    print(z.x)
    print(z.y)

    ec = EdgeClass.HORIZONTAL
    print(ec)

    filename = "/home/bird/dataset/KTH_CampusValhallavagen_Floorplan_Dataset_removeconflicted/" \
               "A0043001/0510034689_Layout1.xml"
    floorplan = FloorPlanGraph()
    floorplan.loadFromXml(filename, rootNodeName="floor")
    image = floorplan.to_image(target_resolution=0.2)

    import matplotlib.pyplot as plt

    plt.imshow(image, cmap='gray')
    plt.show()
    plt.close()

    samples = floorplan.get_samples(target_resolution=0.2, sample_distance=5)
    for i in range(0, len(samples)):
        plt.imshow(samples[i], cmap='gray')
        plt.show()
        plt.close()



