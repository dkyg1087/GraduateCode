import picar_4wd as fc
import a_star
from matplotlib import pyplot as plt 
from math import cos,sin
import numpy as np
import time

sec = False
SIZE = 10
car_dir = 'f'
car_pos = (SIZE*2-1,SIZE)
end_pos = (0,SIZE*2-1) if sec else (0,0)
fig, ax = plt.subplots()

def wall_and_padding(maze,pos):
    temp = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),(0,0)]
    #temp = [(0, -2), (0, 2), (-2, 0), (-2, -2), (-2, 2),(0,0),(2,2),(2,-2)]
    #temp = [(0,0)]
    for direction in temp:
        if pos[0]+direction[0] > SIZE*2-1 or pos[0]+direction[0] < 0 or pos[1]+direction[1] > SIZE*2-1 or pos[1]+direction[1] < 0:
            #print("point passed",pos[0]+direction[0],pos[1]+direction[1])
            continue
        #if maze[pos[0]+direction[0]][pos[1]+direction[1]] == '1':
            #print((pos[0]+direction[0],pos[1]+direction[1]),"already marked")
        maze[pos[0]+direction[0]][pos[1]+direction[1]] = 1
    return

def scan(maze):
    polar = {}
    fc.servo.set_angle(70)
    time.sleep(.5)
    for angle in range(70,-71,-10):
        dis = fc.get_distance_at(angle)
        if dis == -2:
            dis = 1000
        polar[angle] = dis
    time.sleep(.5)
    #print(polar)
    for angle in range(-70,71,10):
        dis = fc.get_distance_at(angle)
        if dis == -2:
            dis = 1000
        #if max(polar[angle],dis)>=55:
            #polar[angle] = 60        
        else:
            polar[angle] = max(polar[angle],dis)
    #print(polar)
    #pi = 3.1415926535
    #theta = [(2*pi/360)*(angle+90) for angle in polar.keys()]
    #ax.plot(theta,polar.values())
    for angle, dist in polar.items():
        if dist >= 45 :
            continue
        x = round((dist*cos(np.radians(angle+90)))/5)
        y = round((dist*sin(np.radians(angle+90)))/5)
        #print(y,x)
        # car pos is (99,50)
        if car_dir == 'f':
            wall_and_padding(maze,(car_pos[0]-y,car_pos[1]+x))
        elif car_dir == 'b':
            wall_and_padding(maze,(car_pos[0]+y,car_pos[1]-x))
        elif car_dir == 'l':
            wall_and_padding(maze,(car_pos[0]-x,car_pos[1]-y))
        else:
            wall_and_padding(maze,(car_pos[0]+x,car_pos[1]+y))

def car_forward():
    global car_dir
    if car_dir == 'f':
        fc.forward(10)
        time.sleep(.2)
        fc.stop()
    elif car_dir == 'b':
        fc.backward(10)
        time.sleep(.2)
        fc.stop()
    elif car_dir == 'l':
        fc.turn_right(30)
        time.sleep(1)
        fc.stop()
        fc.forward(10)
        time.sleep(.2)
        fc.stop()
        car_dir = 'f'
    else:
        fc.turn_left(30)
        time.sleep(1.2)
        fc.stop()
        fc.forward(10)
        time.sleep(.3)
        fc.stop()
        car_dir = 'f'
    return

def car_left():
    global car_dir
    if car_dir == 'f':
        fc.turn_left(30)
        time.sleep(1.2)
        fc.stop()
        fc.forward(10)
        time.sleep(.3)
        fc.stop()
        car_dir = 'l'
    elif car_dir == 'l':
        fc.forward(10)
        time.sleep(.3)
        fc.stop()
    return

def car_backward():
    global car_dir
    if car_dir == 'b':
        fc.forward(10)
        time.sleep(.2)
        fc.stop()
    elif car_dir == 'f':
        fc.backward(10)
        time.sleep(.2)
        fc.stop()
    elif car_dir == 'r':
        fc.turn_right(30)
        time.sleep(.9)
        fc.stop()
        fc.forward(10)
        time.sleep(.2)
        fc.stop()
        car_dir = 'f'
    return

def car_right():
    global car_dir
    if car_dir == 'f':
        fc.turn_right(30)
        time.sleep(.9)
        fc.stop()
        fc.forward(10)
        time.sleep(.3)
        fc.stop()
        car_dir = 'r'
    elif car_dir == 'r':
        fc.forward(10)
        time.sleep(.3)
        fc.stop()
    return

def main():
    global car_pos,end_pos
    goal = False
    maze = [[0]*(SIZE*2) for _ in range(SIZE*2)]
    #maze[end_pos[0]][end_pos[1]]=4
    #scan(maze)
    #maze[car_pos[0]][car_pos[1]]="c"
    #with open('maze.txt','w') as f:
        #for row in maze:
            #f.write("".join(row))
            #f.write("\n")
    #plt.show()
    while not goal:
        #maze = [[' ']*(SIZE*2) for _ in range(SIZE*2)]
        scan(maze)
        #with open('maze.txt','w') as f:
            #for row in maze:
                #f.write("".join(row))
                #f.write("\n")
        try:
            path , direc = a_star.astar(maze,car_pos,end_pos)
        except TypeError:
            car_backward()
            scan(maze)
            path , direc = a_star.astar(maze,car_pos,end_pos)
            if car_dir == 'f':
                car_pos = (carpos[0]+1,car_pos[1])
            elif car_dir == 'r':
                car_pos = (carpos[0],car_pos[1]-1)
            else:
                car_pos = (carpos[0],car_pos[1]+1)
        for i in range(5):
            if direc[i]== 'f':
                car_forward()
            elif direc[i]=='l':
                car_left()
            elif direc[i]=='r':
                car_right()
            else:
                car_backward()
            
            ax.cla()
            ax.pcolor(maze,edgecolors='k',linewidths=1)
            plt.pause(.01)
            
            car_pos = path[i]
            maze[car_pos[0]][car_pos[1]]=3
            if car_pos == end_pos:
                goal = True
                print("GOAL")
                fc.turn_right(60)
                time.sleep(2)
                break
    plt.close()
    return


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        fc.stop()
        plt.close()
    finally:
        fc.stop()
        plt.close()