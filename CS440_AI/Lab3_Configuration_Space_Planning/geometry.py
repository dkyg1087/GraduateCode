# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP3
"""

import math
import numpy as np
from alien import Alien
from typing import List, Tuple

def does_alien_touch_wall(alien, walls,granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    if alien.is_circle():
        point = alien.get_centroid()
        width = alien.get_width()
        for wall in walls:
            dist = point_segment_distance(point,((wall[0],wall[1]),(wall[2],wall[3])))
            if dist < (granularity/(2**0.5))+width or np.isclose(dist,(granularity/(2**0.5))+width):
                return True
        return False
    else:
        points = alien.get_head_and_tail()
        width = alien.get_width()
        for wall in walls:
            dist = segment_distance(points,((wall[0],wall[1]),(wall[2],wall[3])))
            if dist < (granularity/(2**0.5))+width or np.isclose(dist,(granularity/(2**0.5))+width):
                return True
        return False

def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """
    if alien.is_circle():
        point = alien.get_centroid()
        width = alien.get_width()
        for goal in goals:
            dist = ((goal[0]-point[0])**2+(goal[1]-point[1])**2)**0.5
            if dist < goal[2]+ width or np.isclose(dist,goal[2]+ width):
                return True
    else:
        points = alien.get_head_and_tail()
        width = alien.get_width()
        for goal in goals:
            dist = point_segment_distance(goal,points)
            if dist < goal[2]+ width or np.isclose(dist,goal[2]+ width):
                return True
    return False
                
def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    if alien.is_circle():
        point = alien.get_centroid()
        width = alien.get_width()
        if point[0]-width <= 0 or point[0] + width >= window[0] or point[1]-width <= 0 or point[1] + width >= window[1]:
            return False
        return True
    else:
        points = alien.get_head_and_tail()
        width = alien.get_width()
        for wall in [((0,0),(window[0],0)),((0,0),(0,window[1])),((0,window[1]),window),((window[0],0),window)]:
            dist = segment_distance(points,wall)
            if dist < (granularity/(2**0.5))+width or np.isclose(dist,(granularity/(2**0.5))+width):
                return False
        return True

def point_segment_distance(point, segment):
    
    
    x1 = segment[0][0]
    y1 = segment[0][1]
    x2 = segment[1][0]
    y2 = segment[1][1]
    x3 = point[0]
    y3 = point[1]
    px = x2-x1
    py = y2-y1

    if px == 0 and py == 0:
        return ((x3-x1)**2+(y3-y1)**2)**0.5
    norm = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = (dx*dx + dy*dy)**.5
    return dist

def do_segments_intersect(segment1, segment2):

    def onSegment(p, q, r):
        if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
            return True
        return False
  
    def orientation(p, q, r): 
        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
        if (val > 0):
            return 1
        elif (val < 0):
            return 2
        else:
            return 0
    p1 = segment1[0]
    q1 = segment1[1]
    p2 = segment2[0]
    q2 = segment2[1]
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if ((o1 != o2) and (o3 != o4)):
        return True

    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True

    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True

    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True

    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True

    return False

def segment_distance(segment1, segment2):
    
    if do_segments_intersect(segment1, segment2):
        return 0
    else:
        minn = point_segment_distance(segment1[0],segment2)
        minn = min(minn,point_segment_distance(segment1[1],segment2))
        for point in segment2:
            minn = min(minn,point_segment_distance(point,segment1))
    return minn

if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result

    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                  f'{b} is expected to be {result[i]}, but your' \
                                                                  f'result is {distance}'

    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls, 0)
        touch_goal_result = does_alien_touch_goal(alien, goals)
        in_window_result = is_alien_within_window(alien, window, 0)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert touch_goal_result == truths[
            1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {touch_goal_result}, ' \
                f'expected: {truths[1]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    # Initialize Aliens and perform simple sanity check.
    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")