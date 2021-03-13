

def get_status(polygon, point):

    min_x = min([p[0] for p in polygon])
    min_y = min([p[1] for p in polygon])
    max_x = max([p[0] for p in polygon])
    max_y = max([p[1] for p in polygon])

    if min_x < point[0] < max_x and min_y < point[1] < max_y:
        return True
    else:
        return False
