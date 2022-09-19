import picar_4wd as fc
#from statistics import mean


speed = 15
polar = {}
def main():
    while True:
        for angle in range(-30,31,10):
            dis = fc.get_distance_at(angle)
            if dis == -2:
                dis = 60
            polar[angle] = dis
        if any( x < 25 for x in polar.values()):
            fc.turn_right(speed)
        else:
            fc.forward(speed)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fc.stop()