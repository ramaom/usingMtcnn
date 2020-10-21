import math

def getDistanceBetween2Point(point1, point2):
    width=point1[0]-point2[0]
    height=point1[1]-point2[1]
    return math.sqrt((width*width) + (height*height))
