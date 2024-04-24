from enum import IntEnum

class Activity(IntEnum):
    STANDING = 0
    WALKING = 1
    RUNNING = 2
    CYCLING = 3


class WatchLocation(IntEnum):
    WRIST = 0
    BELT = 1
    ANKLE = 2

class Path(IntEnum):
    PATH_0 = 0
    PATH_1 = 1
    PATH_2 = 2
    PATH_3 = 3
    PATH_4 = 4
