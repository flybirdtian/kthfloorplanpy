# try:
#     import xml.etree.cElementTree as ET
# except ImportError:
#     import xml.etree.ElementTree as ET

import xml.etree.cElementTree as ET
import networkx as nx
from FloorPlanGraph import *
import os

class GraphFileOperations(object):
    @classmethod
    def loadGraphFromXML(cls, filenamePath: str, rootNodeName: str) -> (nx.Graph, graphProperties):
        ''' load Graph from xml file
        Read a graph from the xml file, the node in the graph is Space, we use space name to identify a node,
        and use Space to represent its properties (which also contains space name)
        '''

        xml_tree = ET.ElementTree(file=filenamePath)
        outGraph = nx.Graph()

        currentGraphProperty = graphProperties()
        currentGraphProperty.filepath = filenamePath

        path, name = os.path.split(filenamePath)
        currentGraphProperty.floorname = name

        xml_root = xml_tree.getroot()
        if xml_root.tag == rootNodeName:

            # add nodes
            for v in xml_root:
                if v.tag == "space":
                    spaceName = v.attrib["name"]  # type : str
                    spaceType = v.attrib["type"]

                    nameSplit = spaceName.split('.')
                    spaceName = nameSplit[0]

                    # different from c++ code, the spaceName is both space name and currentVertex
                    if spaceName != "":
                        if spaceName not in outGraph.nodes:  # add new nodes
                            outGraph.add_node(spaceName)
                            space = Space()
                            space.category = spaceType
                            space.vertex_id = spaceName
                            outGraph.nodes[spaceName]["attr"] = space

                        # For each spaces's children
                        for i in v:
                            if i.tag == "contour":
                                points = []  # type: [Point2D]
                                # for each contours' children
                                for l in i:
                                    if l.tag == "centroid":
                                        outGraph.nodes[spaceName]["attr"].centroid.x = float(l.attrib["x"])
                                        outGraph.nodes[spaceName]["attr"].centroid.y = float(l.attrib["y"])
                                    elif l.tag == "extent":
                                        outGraph.nodes[spaceName]["attr"].maxx = float(l.attrib["maxx"])
                                        outGraph.nodes[spaceName]["attr"].maxy = float(l.attrib["maxy"])
                                        outGraph.nodes[spaceName]["attr"].minx = float(l.attrib["minx"])
                                        outGraph.nodes[spaceName]["attr"].miny = float(l.attrib["miny"])
                                    elif l.tag == "point":
                                        point = Point2D(x=float(l.attrib["x"]), y=float(l.attrib["y"]))
                                        points.append(point)
                                    elif l.tag == "linesegment":
                                        lseg = LineSegment()
                                        lseg.startPos = Point2D(x=float(l.attrib["x1"]), y=float(l.attrib["y1"]))
                                        lseg.endPos = Point2D(x=float(l.attrib["x2"]), y=float(l.attrib["y2"]))
                                        lseg.type = l.attrib["type"]
                                        lseg.portalToRoom = l.attrib["target"]
                                        outGraph.nodes[spaceName]["attr"].roomLayout.append(lseg)

                                if len(points) > 0:
                                    lseg = LineSegment()
                                    for j in range(0, len(points)):
                                        if j == len(points) - 1:
                                            lseg.startPos = points[j]
                                            lseg.endPos = points[0]
                                        else:
                                            lseg.startPos = points[i]
                                            lseg.endPos = points[i + 1]
                                        outGraph.nodes[spaceName]["attr"].roomLayout.append(lseg)

                                outGraph.nodes[spaceName]["attr"].updateExtent()



                elif v.tag == "Scale":
                    currentGraphProperty.pixel_distance = float(v.attrib["PixelDistance"])
                    currentGraphProperty.real_distance = float(v.attrib["RealDistance"])
            # add edges
            for v in xml_root:
                if v.tag == "space":
                    sourceSpaceName = v.attrib["name"]
                    nameSplit = sourceSpaceName.split('.')
                    sourceSpaceName = nameSplit[0]

                    for i in v:
                        if i.tag == "portal":
                            if "target" in i.attrib:
                                targetSpaceName = i.attrib["target"]
                                nameSplit = targetSpaceName.split('.')
                                targetSpaceName = nameSplit[0]
                                if (sourceSpaceName, targetSpaceName) not in outGraph.edges and \
                                        sourceSpaceName != targetSpaceName:
                                    outGraph.add_edge(sourceSpaceName, targetSpaceName)

        outGraph.graph["attr"] = currentGraphProperty
        return outGraph, currentGraphProperty

    @classmethod
    def loadAllGraphsInFolder(cls, sdir: str, rootNodeName: str) -> ([nx.Graph], [graphProperties]):
        graphs = []
        graphProperties = []

        if not os.path.exists(sdir) or not os.path.isdir(sdir):
            return graphs, graphProperties

        for sub in [os.path.join(sdir, o) for o in os.listdir(sdir)]:
            if os.path.isdir(sub):
                print("Loading directory %s" % sub)
                graphs_tmp, graphProperties_tmp = GraphFileOperations.loadAllGraphsInFolder(sdir=sub,
                                                                                            rootNodeName=rootNodeName)
                graphs.extend(graphs_tmp)
                graphProperties.extend(graphProperties_tmp)
            elif os.path.isfile(sub) and os.path.splitext(sub)[1] == ".xml":
                # check if it is a file and its extension name is .xml
                print("Loading file %s" % sub)
                graph_tmp, props_tmp = GraphFileOperations.loadGraphFromXML(filenamePath=sub, rootNodeName=rootNodeName)
                graphs.append(graph_tmp)
                graphProperties.append(props_tmp)

        return graphs, graphProperties


if __name__ == "__main__":
    filename = "/home/bird/dataset/KTH_CampusValhallavagen_Floorplan_Dataset_removeconflicted/A0043001/0510034689_Layout1.xml"
    tree = ET.ElementTree(file=filename)
    root = tree.getroot()
    print(root.tag)
    attrib_root = root.attrib
    print(attrib_root["BuildingName"])
    print(type(root))
    # for child_of_root in root:
    #     print(child_of_root.tag, child_of_root.attrib)

    graph, graph_property = GraphFileOperations.loadGraphFromXML(filenamePath=filename, rootNodeName="floor")
    import matplotlib.pyplot as plt

    # plt.subplot(121)
    plt.plot()
    nx.draw(graph, with_labels="True", font_weight='bold')
    plt.show()
    plt.close()

    print(dir(graph))

    sdir = "/home/bird/dataset/KTH_CampusValhallavagen_Floorplan_Dataset_removeconflicted"
    graphs, graphProperties = GraphFileOperations.loadAllGraphsInFolder(sdir, rootNodeName="floor")
    for g in graphs:
        nx.draw(g)
        plt.draw()
        plt.show()
